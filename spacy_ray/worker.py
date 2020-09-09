import time
import os
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
from thinc.config import Config
from spacy.cli._util import get_sourced_components
from spacy.cli.train import msg, train_while_improving, load_from_paths
from spacy.cli.train import create_train_batches, create_evaluation_callback
from spacy.cli.train import update_meta
from spacy import util
from spacy.language import Language
from spacy.gold import Corpus
from thinc.api import require_gpu, use_pytorch_for_gpu_memory, Optimizer
from .thinc_proxies import RayHeadProxy, RayChildProxy, RayProxy, RayPeerProxy
from .util import set_params_proxy, make_key


class Worker:
    """Actor class for parallel training."""

    rank: int
    num_workers: int
    gpu_id: int
    nlp: Language
    config: Config
    strategy: str
    _results: List
    _evaluation_callback: Any

    def __init__(
        self,
        config: Config,
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

    def train(self, use_gpu: bool, conn, evaluater, conn_type) -> None:
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

        self._set_params_proxies(self.nlp, conn, conn_type)
        train_batches = create_train_batches(
            self.config["training"]["train_corpus"](self.nlp),
            self.config["training"]["batcher"],
            self.config["training"]["max_epochs"],
        )

        training_step_iterator = train_while_improving(
            self.nlp,
            FakeOptimizer(conn, self.rank),
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
        for batch, info, is_best_checkpoint in training_step_iterator:
            if self.rank == 0 and is_best_checkpoint is not None:
                info["words"] *= self.num_workers
                print_row(info)

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

    def _set_params_proxies(self, nlp: Language, conn, strategy) -> None:
        if strategy == "shared_optimizer":
            proxy = RayProxy(conn, ray=self.ray, use_thread=True)
        elif strategy == "peer_params":
            proxy = RayPeerProxy(
                conn, self.get_optimizer(), self.get_owned_keys(nlp), ray=self.ray
            )
        else:
            if self.rank == 0:
                proxy = RayHeadProxy(
                    conn, self.get_optimizer(), self.get_quorum(), ray=self.ray
                )  # type: ignore
            else:
                proxy = RayChildProxy(conn)  # type: ignore

        for name, component in nlp.pipeline:
            if hasattr(component, "model"):
                set_params_proxy(component.model, proxy)

    def get_owned_keys(self):
        owned_keys = []
        for name, component in self.nlp.pipeline:
            if not hasattr(component, "model"):
                continue
            for node in component.model.walk():
                if (node.id % self.num_workers) == self.rank:
                    for param_name in node.param_names:
                        owned_keys.append(make_key(node.id, param_name))
        print("Owned keys", self.rank, owned_keys)
        return owned_keys


class FakeOptimizer:
    def __init__(self, conn, worker_id):
        self.conn = conn
        self.worker_id = worker_id
        self._futures = []
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
        # ray.get(self._futures)
        # self._futures = []
        # if self.worker_id == 0:
        #    self._futures.append(self.conn.step_schedules.remote())
        # self._futures.append(self.conn.inc_progress.remote(self.worker_id))


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
