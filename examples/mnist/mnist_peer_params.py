"""
PyTorch version: https://github.com/pytorch/examples/blob/master/mnist/main.py
TensorFlow version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
"""
# pip install thinc ml_datasets typer
import threading
from typing import Optional
import time
from thinc.api import Model, chain, Relu, Softmax, Adam
from thinc.types import FloatsXd
import ml_datasets
from wasabi import msg
from tqdm import tqdm
import typer
from spacy_ray.thinc_proxies import RayPeerProxy
from spacy_ray.util import set_params_proxy, divide_params
import ray

class Timer:
    def __init__(self, state):
        self.state = state
        self.sum = 0
        self.n = 0

    def __enter__(self):
        self.start = time.time()
        self.n += 1

    def __exit__(self, *args):
        interval = time.time() - self.start
        self.sum += interval
        print(f"{self.state}: {self.sum / self.n:0.4f}")


def thread_training(train_data, model):
    for X, Y in train_data:
        Yh, backprop = model.begin_update(X)
        backprop(Yh - Y)


@ray.remote(num_cpus=2)
class Worker:
    def __init__(self, i, n_workers):
        self.i = i
        self.n_workers = n_workers
        self.optimizer = Adam(0.001)
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.timers = {k: Timer(k) for k in ["forward", "backprop"]}
        self.thread = None

    def sync_params(self):
        for key in self.proxy._owned_keys:
            self.proxy.send_param(key)

    def inc_grad(self, key, version, value) -> None:
        assert key in self.proxy._owned_keys
        if self.proxy.check_version(key, version):
            self.proxy.inc_grad(key[0], key[1], value)
    
    def set_param(self, key, version, value) -> Optional[FloatsXd]:
        return self.proxy.receive_param(key, version, value)

    def add_model(self, n_hidden, dropout):
        # Define the model
        self.model = chain(
            Relu(nO=n_hidden, dropout=dropout),
            Relu(nO=n_hidden, dropout=dropout),
            Softmax(),
        )
    
    def add_data(self, batch_size):
        # Load the data
        (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
        shard_size = len(train_X) // self.n_workers
        shard_start = self.i * shard_size
        shard_end = shard_start + shard_size
        self.train_data = self.model.ops.multibatch(
            batch_size,
            train_X[shard_start : shard_end],
            train_Y[shard_start : shard_end],
            shuffle=True
        )
        self.dev_data = self.model.ops.multibatch(batch_size, dev_X, dev_Y)
        # Set any missing shapes for the model.
        self.model.initialize(X=train_X[:5], Y=train_Y[:5])
        self.sync_params()

    def set_proxy(self, workers, optimizer):
        worker_keys = divide_params(self.model, self.n_workers)
        peer_map = {}
        for peer, keys in zip(workers, worker_keys):
            for key in keys:
                peer_map[key] = peer
        self.proxy = RayPeerProxy(peer_map, optimizer, worker_keys[self.i])
        set_params_proxy(
            self.model,
            self.proxy
        )

    def train_epoch(self):
        # Really not sure whether this works.
        # Try running the actual work in a child thread, so we're not blocking
        # the thread and can receive from the other workers.
        self.thread = threading.Thread(
            target=thread_training,
            args=(self.train_data, self.model)
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


def main(
    n_hidden: int = 256, dropout: float = 0.2, n_iter: int = 10, batch_size: int = 256,
    n_epoch: int=10, quorum: int=None, n_workers: int=2, use_thread: bool = False
):
    if quorum is None:
        quorum = n_workers
    batch_size //= n_workers
    ray.init(lru_evict=True)
    workers = []
    optimizer = Adam(0.001)
    print("Add workers and model")
    for i in range(n_workers):
        worker = Worker.remote(i, n_workers)
        ray.get(worker.add_model.remote(n_hidden, dropout))
        workers.append(worker)
    print("Set proxy")
    for worker in workers:
        ray.get(worker.set_proxy.remote(workers, optimizer))
    print("Set data")
    for worker in workers:
        ray.get(worker.add_data.remote(batch_size))

    print("Train")
    for i in range(n_epoch):
        with Timer("epoch"):
            for worker in workers:
                ray.get(worker.train_epoch.remote())
            todo = list(workers)
            while todo:
                time.sleep(1)
                todo = [w for w in workers if ray.get(w.is_running.remote())]
        print(i, ray.get(workers[0].evaluate.remote()))


if __name__ == "__main__":
    typer.run(main)
