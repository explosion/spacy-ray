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



@ray.remote
class Worker:
    def __init__(self, i, n_workers):
        self.i = i
        self.n_workers = n_workers
        self.optimizer = Adam(0.001)
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.timers = {k: Timer(k) for k in ["forward", "backprop"]}

    def sync_params(self):
        for key in self.proxy._owned_keys:
            self.proxy.send_param(key)

    async def inc_grad(self, key, version, value) -> None:
        assert key in self.proxy._owned_keys
        if self.proxy.check_version(key, version):
            self.proxy.inc_grad(key[0], key[1], value)
    
    async def set_param(self, key, version, value) -> Optional[FloatsXd]:
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
        self.train_data = list(self.model.ops.multibatch(
            batch_size,
            train_X[shard_start : shard_end],
            train_Y[shard_start : shard_end],
            shuffle=True
        ))
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

    async def train_epoch(self):
        for i in range(len(self.train_data)):
            await self.train_batch(i)

    async def train_batch(self, index):
        if index >= len(self.train_data):
            return None
        else:
            X, Y = self.train_data[index]
            Yh, backprop = self.model.begin_update(X)
            backprop(Yh - Y)
            return (self.i, index+1)

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
    for i in range(n_workers):
        worker = Worker.options(max_concurrency=2).remote(i, n_workers)
        ray.get(worker.add_model.remote(n_hidden, dropout))
        workers.append(worker)
    for worker in workers:
        ray.get(worker.set_proxy.remote(workers, optimizer))
    for worker in workers:
        ray.get(worker.add_data.remote(batch_size))

    for i in range(n_epoch):
        with Timer("epoch"):
            todo = [w.train_batch.remote(0) for w in workers]
            while todo:
                done, next_todo = ray.wait(todo, num_returns=1)
                for result in ray.get(done):
                    if result is not None:
                        worker_i, data_i = result
                        next_todo.append(workers[worker_i].train_batch.remote(data_i))
                todo = next_todo
            print(i, ray.get(workers[0].evaluate.remote()))


if __name__ == "__main__":
    typer.run(main)
