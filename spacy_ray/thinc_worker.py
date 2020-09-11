"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
import threading
from typing import Optional
from thinc.types import FloatsXd
import ml_datasets
from thinc.api import registry, get_current_ops, Config
from spacy_ray.proxies import RayPeerProxy
from spacy_ray.util import set_params_proxy, divide_params


def thread_training(train_data, model):
    for X, Y in train_data:
        Yh, backprop = model.begin_update(X)
        backprop(Yh - Y)


class ThincWorker:
    """Worker for training Thinc models with Ray.

    Mostly used for development, e.g. for the mnist scripts.
    """

    def __init__(
        self,
        config: Config,
        *,
        rank: int = 0,
        num_workers: int = 1,
        ray=None,
    ):
        if ray is None:
            # Avoid importing ray in the module. This allows a test-ray to
            # be passed in, and speeds up the CLI.
            import ray  # type: ignore

            self.ray = ray
        self.rank = rank
        self.num_workers = num_workers
        config = registry.make_from_config(config)
        self.optimizer = config["optimizer"]
        self.train_data = config["train_data"]
        self.dev_data = config["dev_data"]
        self.thread = None
        self.proxy = None
        self.n_grads_used = 0
        self.n_grads_discarded = 0

    def get_percent_grads_used(self):
        total = self.n_grads_used + self.n_grads_discarded
        if total == 0:
            return None
        else:
            return self.n_grads_used / total

    def add_model(self, model):
        self.model = model
        for X, Y in self.train_data:
            self.model.initialize(X=X, Y=Y)
            break

    def sync_params(self):
        for key in self.proxy._owned_keys:
            self.proxy.send_param(key)

    def inc_grad(self, key, version, value) -> None:
        assert key in self.proxy._owned_keys
        if self.proxy.check_version(key, version):
            self.proxy.inc_grad(key[0], key[1], value)
            self.n_grads_used += 1
        else:
            self.n_grads_discarded += 1

    def set_param(self, key, version, value) -> Optional[FloatsXd]:
        return self.proxy.receive_param(key, version, value)

    def set_proxy(self, workers, quorum):
        worker_keys = divide_params(self.model, self.num_workers)
        peer_map = {}
        for peer, keys in zip(workers, worker_keys):
            for key in keys:
                peer_map[key] = peer
        self.proxy = RayPeerProxy(
            peer_map,
            self.optimizer,
            worker_keys[self.rank],
        )
        set_params_proxy(self.model, self.proxy)

    def train_epoch(self):
        self.thread = threading.Thread(
            target=thread_training, args=(self.train_data, self.model)
        )
        self.thread.start()

    def is_running(self):
        return self.thread.is_alive()

    def evaluate(self):
        correct = 0
        total = 0
        for X, Y in self.dev_data:
            Yh = self.model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        return correct / total


@registry.datasets("mnist_train_batches.v1")
def get_train_data(worker_id, num_workers, batch_size):
    ops = get_current_ops()
    # Load the data
    (train_X, train_Y), _ = ml_datasets.mnist()
    shard_size = len(train_X) // num_workers
    shard_start = worker_id * shard_size
    shard_end = shard_start + shard_size
    return list(
        ops.multibatch(
            batch_size,
            train_X[shard_start:shard_end],
            train_Y[shard_start:shard_end],
            shuffle=True,
        )
    )


@registry.datasets("mnist_dev_batches.v1")
def get_dev_data(batch_size):
    ops = get_current_ops()
    _, (dev_X, dev_Y) = ml_datasets.mnist()
    dev_data = ops.multibatch(batch_size, dev_X, dev_Y)
    return list(dev_data)
