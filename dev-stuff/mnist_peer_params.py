import copy
import typer
import ray
import time
from timeit import default_timer as timer
from datetime import timedelta
import ml_datasets
from thinc_worker import ThincWorker
from thinc.api import Model
from thinc.types import Floats2d

# This config data is passed into the workers, so that they can then
# create the objects.

CONFIG = {
    "optimizer": {
        "@optimizers": "Adam.v1",
        "learn_rate": 0.001
    },
    "train_data": {
        "@datasets": "mnist_train_batches.v1",
        "worker_id": None,
        "num_workers": None,
        "batch_size": None
    },
    "dev_data": {
        "@datasets": "mnist_dev_batches.v1",
        "batch_size": None
    }
}

def make_model(n_hidden: int, depth: int, dropout: float) -> Model[Floats2d, Floats2d]:
    from thinc.api import chain, clone, Relu, Softmax
    return chain(
        clone(Relu(nO=n_hidden, dropout=dropout), depth),
        Softmax()
    )


def main(
    n_hidden: int = 256,
    depth: int = 2,
    dropout: float = 0.2,
    n_iter: int = 10,
    batch_size: int = 64,
    n_epoch: int=10,
    quorum: int=1,
    n_workers: int=2
):
    model = make_model(n_hidden, depth, dropout)
    CONFIG["train_data"]["batch_size"] = batch_size
    CONFIG["dev_data"]["batch_size"] = batch_size
    if quorum is None:
        quorum = n_workers
    ray.init(lru_evict=True)
    workers = []
    print("Add workers and model")
    Worker = ray.remote(ThincWorker)
    for i in range(n_workers):
        config = copy.deepcopy(CONFIG)
        config["train_data"]["worker_id"] = i
        config["train_data"]["num_workers"] = n_workers
        worker = Worker.remote(
            config,
            rank=i,
            num_workers=n_workers,
            ray=ray
        )
        ray.get(worker.add_model.remote(model))
        workers.append(worker)
    for worker in workers:
        ray.get(worker.set_proxy.remote(workers, quorum))
    for worker in workers:
        ray.get(worker.sync_params.remote())
    print("Train")
    for i in range(n_epoch):
        start = timer()
        for worker in workers:
            ray.get(worker.train_epoch.remote())
        todo = list(workers)
        while todo:
            time.sleep(1)
            todo = [w for w in workers if ray.get(w.is_running.remote())]
        end = timer()
        duration = timedelta(seconds=int(end - start))
        grads_usage = [ray.get(w.get_percent_grads_used.remote()) for w in workers]
        print(duration, i, ray.get(workers[0].evaluate.remote()), grads_usage)


if __name__ == "__main__":
    typer.run(main)
