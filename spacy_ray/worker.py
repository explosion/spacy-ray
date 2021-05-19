from typing import List, Dict, Union, Any, Optional
import time
import os
import threading
from pathlib import Path
from thinc.config import Config
from thinc.types import FloatsXd
from spacy.cli._util import import_code
from spacy.training.loop import train_while_improving, create_train_batches
from spacy.training.loop import create_evaluation_callback
from spacy.training.loop import create_before_to_disk_callback
from spacy.training.loop import update_meta
from spacy.training.initialize import init_nlp
from spacy.language import Language
from spacy.util import registry, logger, resolve_dot_names
from spacy.schemas import ConfigSchemaTraining
from thinc.api import require_gpu, set_gpu_allocator

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
    config: Union[Dict[str, Any], Config]
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
        code_path: Optional[Path] = None,
        ray=None,
    ):
        if ray is None:
            # Avoid importing ray in the module. This allows a test-ray to
            # be passed in, and speeds up the CLI.
            import ray  # type: ignore

            self.ray = ray
        import_code(code_path)
        self.rank = rank
        self.num_workers = num_workers
        self.gpu_id = self._resolve_gpu(use_gpu)
        self.nlp = init_nlp(Config(config), use_gpu=self.gpu_id)
        config = self.nlp.config.interpolate()
        self.T = registry.resolve(config["training"], schema=ConfigSchemaTraining)
        dot_names = [self.T["train_corpus"], self.T["dev_corpus"]]
        self.train_corpus, self.dev_corpus = resolve_dot_names(config, dot_names)
        self.before_to_disk = create_before_to_disk_callback(self.T["before_to_disk"])
        allocator = self.T["gpu_allocator"]
        if use_gpu >= 0 and allocator:
            set_gpu_allocator(allocator)
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

    def get_quorum(self) -> int:
        # Default to setting the 'quorum' to be the number of workers multiplied
        # by the accumulate_gradient value. This is how many gradients for a
        # parameter we will accumulate before running the optimizer.
        return self.num_workers * self.T["accumulate_gradient"]

    def train(self, peers: List, evaluator: "Evaluator") -> None:
        def evaluate():
            if self.rank == 0:
                scores = self.evaluate()
                self.ray.get(evaluator.set_scores.remote(scores))
                return scores
            else:
                scores = None
                while scores is None:
                    time.sleep(5)
                    scores = self.ray.get(evaluator.get_scores.remote())
                return scores

        train_batches = create_train_batches(
            self.nlp,
            self.train_corpus,
            self.T["batcher"],
            self.T["max_epochs"],
        )
        training_step_iterator = train_while_improving(
            self.nlp,
            FakeOptimizer(),
            train_batches,
            evaluate=evaluate,
            dropout=self.T["dropout"],
            accumulate_gradient=1,
            patience=self.T["patience"],
            max_steps=self.T["max_steps"],
            eval_frequency=self.T["eval_frequency"],
            exclude=self.T["frozen_components"],
        )
        if self.rank == 0:
            print_row, finalize_logger = self.T["logger"](self.nlp)
        else:
            print_row = lambda: None
        self.thread = threading.Thread(
            target=thread_training,
            args=(
                training_step_iterator,
                print_row,
                self.rank,
                self.num_workers,
                self.gpu_id,
            ),
        )
        self.thread.start()

    def is_running(self):
        return self.thread.is_alive()

    def evaluate(self) -> Dict[str, Union[Dict[str, float], float]]:
        if not self._has_evaluation_callback:
            self._evaluation_callback = create_evaluation_callback(
                self.nlp,
                self.dev_corpus,
                self.T["score_weights"],
            )
            self._has_evaluation_callback = True
        return self._evaluation_callback()

    def save_checkpoint(self, info: Dict, output_path: Path) -> None:
        with self.nlp.select_pipes(disable=self.T["frozen_components"]):
            update_meta(self.T, self.nlp, info)
        self.before_to_disk(self.nlp).to_disk(output_path)

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
            self.T["optimizer"],
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
            logger.info(f"Using GPU (isolated): {gpu_id}")
            require_gpu(0)
        else:
            logger.info("Using CPU")
            gpu_id = -1
        return gpu_id


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


class Evaluator:
    """Share evaluation results between workers.

    One worker should publish evaluation results to the evaluator,
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
