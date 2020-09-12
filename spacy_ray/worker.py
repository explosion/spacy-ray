import time
import os
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Optional
from thinc.config import Config
from thinc.types import FloatsXd
from spacy.cli._util import get_sourced_components
from spacy.cli.train import msg, train_while_improving, load_from_paths
from spacy.cli.train import create_train_batches, create_evaluation_callback
from spacy.cli.train import update_meta
from spacy import util
from spacy.language import Language
from spacy.training import Corpus
from thinc.api import require_gpu, use_pytorch_for_gpu_memory, Optimizer, get_current_ops
from .proxies import RayPeerProxy
from .util import set_params_proxy, divide_params, KeyT


class Worker:
    """Actor class for spaCy parallel training.

    Okay this is pretty mind-bending stuff...But the idea is that the remote
    workers need to communicate directly, to avoid extra copies. The mechanics
    of this are super twisted though, because it mixes all sorts of levels. But
    it has the be *exactly this* worker object that is passed through, because
    we need the remote actor. That's why it's so strange.

    The workers install a "proxy" object into the Thinc models. When this object
    is installed, Thinc will direct calls to get_param, inc_grad, set_param etc
    through the proxy.

    On each worker, a subset of the weights will be "local". The proxy will
    receive a list of the keys that are local, and a mapping of keys to workers.

    Workers optimize the parameters that are 'local' to them, and then push
    the updates to the other workers. For parameters that aren't local, they
    find the worker that owns that parameter, and publish the gradient to it.

    This strategy is non-blocking, because the gradients and the parameters
    are both pushed, not pulled.

    In order to make this work, we need some concurrency within the workers,
    because the workers need to be listening for updates while continuing to
    work. Currently this is implemented by putting the main training work
    on a thread, and letting the main thread continue to listen for connections.

    Finally, not that there's a pretty tangled circular reference here. I hate
    circular references, it makes the code hard to understand and makes
    Python use GC. But the circular reference here is necessary:

    * Workers hold a reference to the nlp object. Within the nlp object, models
        hold references to the "proxy" object.
    * The proxy object holds a reference to the peer mapping, whose values are
        the workers.
    """

    rank: int
    num_workers: int
    gpu_id: int
    nlp: Language
    config: Config
    proxy: Optional[RayPeerProxy]
    thread: Optional[threading.Thread]
    _results: List
    _evaluation_callback: Any

    def __init__(
        self,
        config: Config,
        *,
        rank: int = 0,
        num_workers: int = 1,
        use_gpu: int = 0,
        ray=None,
    ):
        if ray is None:
            # Avoid importing ray in the module. This allows a test-ray to
            # be passed in, and speeds up the CLI.
            import ray  # type: ignore

            self.ray = ray
        self.rank = rank
        self.num_workers = num_workers
        self.gpu_id = self._resolve_gpu(use_gpu)
        self.nlp, self.config = self._load_nlp_and_config(config)
        self._initialize_models(self.nlp, self.config)
        self._evaluation_callback = lambda: {}
        self._results = []
        self._has_evaluation_callback = False
        self.thread = None
        self.proxy = None
        self.n_grads_used = 0
        self.n_grads_discarded = 0

    ########################################################################
    # Inter-worker communication
    #
    # It'd be nice to have this stuff in a different object, but we need
    # to pass the actual 'actor' handle around, we can't use a shared reference.
    # And if we made another actor, it would run within a different process.
    #
    #########################################################################

    def inc_grad(self, key: KeyT, version: int, value: FloatsXd) -> None:
        if self.proxy is None:
            raise ValueError("Proxy object not set")
        if self.proxy.check_version(key, version):
            self.proxy.inc_grad(key[0], key[1], value)

    def set_param(self, key: KeyT, version: int, value: FloatsXd) -> Optional[FloatsXd]:
        return self.proxy.receive_param(key, version, value)

    def get_param(self, key: KeyT, version: int) -> Optional[FloatsXd]:
        if self.proxy is None:
            raise ValueError("Proxy object not set")
        elif self.proxy.check_version(key, version):
            return self.proxy.get_param(key[0], key[1])
        else:
            return None

    #########################################################################
    # Process control. These are used by the script or function coordinating
    # the work.
    #
    ########################################################################

    def sync_params(self):
        for key in self.proxy._owned_keys:
            self.proxy.send_param(key)

    def get_percent_grads_used(self):
        total = self.n_grads_used + self.n_grads_discarded
        if total == 0:
            return None
        else:
            return self.n_grads_used / total

    def get_optimizer(self) -> Optimizer:
        return self.config["training"]["optimizer"]

    def get_train_corpus(self) -> Corpus:
        return self.config["training"]["train_corpus"]

    def get_dev_corpus(self):
        return self.config["training"]["dev_corpus"]

    def get_quorum(self) -> int:
        # Default to setting the 'quorum' to be the number of workers multiplied
        # by the accumulate_gradient value. This is how many gradients for a
        # parameter we will accumulate before running the optimizer.
        return self.num_workers * self.config["training"]["accumulate_gradient"]

    def train(self, peers: List, evaluater) -> None:
        def evaluate():
            if self.rank == 0:
                scores = self.evaluate()
                self.ray.get(evaluater.set_scores.remote(scores))
                return scores
            else:
                scores = None
                while scores is None:
                    time.sleep(5)
                    scores = self.ray.get(evaluater.get_scores.remote())
                return scores

        train_batches = create_train_batches(
            self.config["training"]["train_corpus"](self.nlp),
            self.config["training"]["batcher"],
            self.config["training"]["max_epochs"],
        )

        training_step_iterator = train_while_improving(
            self.nlp,
            FakeOptimizer(),
            train_batches,
            evaluate=evaluate,
            dropout=self.config["training"]["dropout"],
            accumulate_gradient=1,
            patience=self.config["training"].get("patience", 0),
            max_steps=self.config["training"].get("max_steps", 0),
            eval_frequency=self.config["training"]["eval_frequency"],
            raw_text=None,
            exclude=[],
        )
        if self.rank == 0:
            print_row, finalize_logger = self.config["training"]["logger"](self.nlp)
        else:
            print_row = lambda: None
        self.thread = threading.Thread(
            target=thread_training,
            args=(
                training_step_iterator,
                print_row,
                self.rank,
                self.num_workers,
                self.gpu_id
            )
        )
        self.thread.start()
    
    def is_running(self):
        return self.thread.is_alive()

    def evaluate(self) -> Dict[str, Union[Dict[str, float], float]]:
        if not self._has_evaluation_callback:
            self._evaluation_callback = create_evaluation_callback(
                self.nlp,
                self.get_dev_corpus(),
                self.config["training"]["score_weights"],
            )
            self._has_evaluation_callback = True
        return self._evaluation_callback()

    def save_checkpoint(self, info: Dict, output_path: Path) -> None:
        update_meta(self.config["training"], self.nlp, info)
        self.nlp.to_disk(output_path)

    def get_training_config(self) -> Config:
        return self.config["training"]

    def get_owned_keys(self):
        owned_keys = []
        for name, component in self.nlp.pipeline:
            if hasattr(component, "model"):
                worker_keys = divide_params(component.model, self.num_workers)
                owned_keys.extend(worker_keys[self.rank])
        return owned_keys

    def get_peer_map(self, workers):
        peer_map = {}
        for name, component in self.nlp.pipeline:
            if hasattr(component, "model"):
                worker_keys = divide_params(component.model, self.num_workers)
                for worker, keys in zip(workers, worker_keys):
                    for key in keys:
                        peer_map[key] = worker
        return peer_map

    def set_proxy(self, peers) -> None:
        proxy = RayPeerProxy(
            self.get_peer_map(peers),
            self.get_optimizer(),
            self.get_owned_keys(),
            ray=self.ray,
        )
        for name, component in self.nlp.pipeline:
            if hasattr(component, "model"):
                set_params_proxy(component.model, proxy)
        self.proxy = proxy

    def _resolve_gpu(self, use_gpu: int) -> int:
        if use_gpu >= 0:
            gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", -1))
            msg.info(f"Using GPU (isolated): {gpu_id}")
            require_gpu(0)
        else:
            msg.info("Using CPU")
            gpu_id = -1
        return gpu_id

    def _load_nlp_and_config(self, config: Config) -> Tuple[Language, Config]:
        if config.get("training", {}).get("seed") is not None:
            util.fix_random_seed(config["training"]["seed"])
        if config.get("system", {}).get("use_pytorch_for_gpu_memory"):
            # It feels kind of weird to not have a default for this.
            use_pytorch_for_gpu_memory()
        nlp, config = util.load_model_from_config(config)
        if config["training"]["vectors"] is not None:
            util.load_vectors_into_model(nlp, config["training"]["vectors"])
        return nlp, config

    def _initialize_models(self, nlp: Language, config: Config) -> None:
        optimizer = config["training"]["optimizer"]
        # Components that shouldn't be updated during training
        frozen_components = config["training"]["frozen_components"]
        # Sourced components that require resume_training
        sourced_components = get_sourced_components(config)
        resume_components = [
            p for p in sourced_components if p not in frozen_components
        ]
        if resume_components:
            with nlp.select_pipes(enable=resume_components):
                nlp.resume_training(sgd=optimizer)

        corpus = self.get_train_corpus()
        train_examples = list(corpus(nlp))
        with nlp.select_pipes(disable=[*frozen_components, *resume_components]):
            nlp.begin_training(lambda: train_examples)

        raw_text, tag_map, morph_rules, weights_data = load_from_paths(config)
        if tag_map:
            # Replace tag map with provided mapping
            nlp.vocab.morphology.load_tag_map(tag_map)
        if morph_rules:
            # Load morph rules
            nlp.vocab.morphology.load_morph_exceptions(morph_rules)
        if weights_data is not None:
            raise NotImplementedError


class FakeOptimizer:
    def __init__(self):
        self.averages = {}

    def __call__(self, key, weights, gradient):
        # This shouldn't be called, because when we have the parameter proxy
        # installed, the gradients should never appear, and the `has_grad`
        # check in `model.finish_update` should return False.
        # However, it's difficult to guarantee that for all subclasses and shims
        # so it's safer to noop instead of raising.
        return weights, gradient

    def step_schedules(self):
        pass


class Evaluater:
    """Share evaluation results between workers.

    One worker should publish evaluation results to the Evaluater,
    while the other workers should retrieve them (using a wait-loop if
    necessary).
    """

    def __init__(self):
        self.scores = []

    def set_scores(self, scores):
        self.scores.append(scores)
        return scores

    def get_scores(self):
        if not self.scores:
            return None
        else:
            return self.scores[-1]


def thread_training(training_step_iterator, print_row, rank, num_workers, gpu_id):
    if gpu_id >= 0:
        # I don't fully understand why we need to do this within the thread.
        # I think 0 is also correct here, because ray sets the available devices?
        require_gpu(0)
    for batch, info, is_best_checkpoint in training_step_iterator:
        if rank == 0 and is_best_checkpoint is not None:
            info["words"] *= num_workers
            print_row(info)


