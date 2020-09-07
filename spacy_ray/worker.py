from spacy.cli._util import get_sourced_components
from spacy.cli.train import load_nlp_and_config, msg, train_while_improving
from spacy.cli.train import create_train_batches, create_evaluation_callback
from spacy.cli.train import setup_printer
from thinc.api import require_gpu, use_pytorch_for_gpu_memory
from .thinc_remote_params import RayHeadProxy, RayChildProxy, SharedParams


class Worker:
    def __init__(self, rank, num_workers, use_gpu, config_path, config_overrides):
        self.rank = rank
        self.num_workers = num_workers
        self.gpu_id = self._resolve_gpu(use_gpu)
        self.output_path = output_path
        # Use original config here before it's resolved to functions
        sourced_components = get_sourced_components(config)
        self.nlp, self.config = self._load_nlp_and_config(config_path, config_overrides)
        self._initialize_models(self.nlp, self.config)
        self._evaluation_callback = None
        self._results = []

    def get_optimizer(self):
        return self.config["training"]["optimizer"]

    def get_quorum(self):
        # Default to setting the 'quorum' to be the number of workers multiplied
        # by the accumulate_gradient value. This is how many gradients for a
        # parameter we will accumulate before running the optimizer.
        return self.num_workers * self.config["training"]["accumulate_gradient"]

    def train(self, use_gpu, conn, evaluater):
        def evaluate():
            if self.rank == 0:
                scores = self.evaluate()
                ray.get(evaluater.set_scores.remote(scores))
                return scores
            else:
                scores = None
                while scores is None:
                    time.sleep(5)
                    scores = ray.get(evaluater.get_scores.remote())
                return scores

        self._set_params_proxies(self.nlp, conn)
        train_batches = create_train_batches(
            self.config["train_corpus"](nlp),
            self.config["training"]["batcher"],
            self.config["training"]["max_epochs"],
            self.rank
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
        )
        if self.rank == 0:
            print_row = setup_printer(self.config["training"], self.nlp.pipe_names)
        output_path = self.output_path
        for batch, info, is_best_checkpoint in training_step_iterator:
            if self.rank == 0 and is_best_checkpoint is not None:
                info["words"] *= self.num_workers
                print_row(info)
                if is_best_checkpoint and output_path is not None:
                    self.save_checkpoint(info, output_path / "model-best")

    def evaluate(self):
        if self._evaluation_callback is None:
            self._evaluation_callback = create_evaluation_callback(
                self.nlp,
                self.config["training"]["optimizer"],
                self.corpus,
                self.config["training"]
            )
        return self._evaluation_callback()

    def save_checkpoint(self, info, output_path):
        update_meta(self.config["training"], self.nlp, info)
        self.nlp.to_disk(output_path)

    def get_training_config(self):
        return self.config["training"]

    def _resolve_gpu(self, use_gpu):
        if use_gpu >= 0:
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
            msg.info(f"Using GPU (isolated): {gpu_id}")
            require_gpu(0)
        else:
            msg.info("Using CPU")
            gpu_id = -1
        return gpu_id

    def _load_nlp_and_config(self, config_path: Path, config_overrides):
        config = util.load_config(
            config_path, overrides=config_overrides, interpolate=True
        )
        if config.get("training", {}).get("seed") is not None:
            fix_random_seed(config["training"]["seed"])
        if config.get("system", {}).get("use_pytorch_for_gpu_memory"):
            # It feels kind of weird to not have a default for this.
            use_pytorch_for_gpu_memory()
        nlp, config = util.load_model_from_config(config)
        if config["training"]["vectors"] is not None:
            util.load_vectors_into_model(nlp, config["training"]["vectors"])
        return nlp, config

    def _initialize_models(self, nlp, config, sourced_components):
        optimizer = config["training"]["optimizer"]
        # Components that shouldn't be updated during training
        frozen_components = config["training"]["frozen_components"]
        # Sourced components that require resume_training
        resume_components = [p for p in sourced_components if p not in frozen_components]
        if resume_components:
            with nlp.select_pipes(enable=resume_components):
                nlp.resume_training(sgd=optimizer)
 
        train_examples = list(
            corpus.train_dataset(
                nlp,
                shuffle=False,
                gold_preproc=config["training"]["gold_preproc"],
                max_length=config["training"]["max_length"],
            )
        )
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

    def _set_params_proxies(self, nlp, conn):
        if self.rank == 0:
            proxy = RayHeadProxy(
                conn,
                self.get_optimizer(),
                self.get_quorum(),
                ray=ray
            )
        else:
            proxy = RayChildProxy(conn)
        for name, component in nlp.pipeline:
            if hasattr(component, "model"):
                component.model.set_params_proxy(proxy)


class FakeOptimizer:
    def __init__(self, conn, worker_id):
        self.conn = conn
        self.worker_id = worker_id
        self._futures = []

    def __call__(self, key, weights, gradient):
        raise ValueError("Should not be called?")

    def step_schedules(self):
        pass
        #ray.get(self._futures)
        #self._futures = []
        #if self.worker_id == 0:
        #    self._futures.append(self.conn.step_schedules.remote())
        #self._futures.append(self.conn.inc_progress.remote(self.worker_id))


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
