import asyncio
import ray
import os
import tqdm

from wasabi import msg
from spacy import util
from spacy.cli.train import train
from spacy_ray.param_server import RayOptimizer, RAY_PS_WORKER_GPU_RESERVE


@ray.remote(num_cpus=0)
class ProgressReporter:
    def __init__(self):
        self.updated = asyncio.Event()

    async def get_update_blocking(self):
        await self.updated.wait()
        self.updated.clear()
        return 1

    def update(self):
        self.updated.set()

def setup_and_train(use_gpu, train_args, rank):
    if use_gpu >= 0:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        msg.info(f"Using GPU (isolated): {gpu_id}")
        util.use_gpu(0)
    else:
        msg.info("Using CPU")
    if rank != 0:
        train_args["disable_tqdm"] = True
    train(randomization_index=rank, **train_args)


def print_tqdm(reporter, eval_freq):
    while True:
        progress = tqdm.tqdm(total=eval_freq, leave=False)
        for i in range(eval_freq):
            ray.get(reporter.get_update_blocking.remote())
            progress.update(1)
        progress.close()


def distributed_setup_and_train(use_gpu, num_workers, strategy, ray_address, train_args):
    config_path = train_args["config_path"]
    if ray_address is not None:
        ray.init(address=ray_address)
    else:
        ray.init(ignore_reinit_error=True)
    if strategy == "ps":
        remote_train = ray.remote(setup_and_train)
        if use_gpu >= 0:

            msg.info("Enabling GPU with Ray")
            remote_train = remote_train.options(
                num_gpus=RAY_PS_WORKER_GPU_RESERVE)

        train_args["remote_optimizer"] = RayOptimizer(
            config_path, use_gpu=use_gpu, world_size=num_workers)
        ray.get([
            remote_train.remote(use_gpu, train_args, rank=rank)
            for rank in range(num_workers)
        ])
    elif strategy == "allreduce":
        assert use_gpu >= 0, "All-reduce strategy can only be used with GPU!"
        from spacy_ray.optimizer import RayNCCLWorker, AllreduceOptimizer

        reporter = ProgressReporter.remote()
        RemoteRayWorker = ray.remote(RayNCCLWorker).options(num_gpus=1)
        workers = [
            RemoteRayWorker.remote(rank, num_workers)
            for rank in range(num_workers)
        ]
        head_id = ray.get(workers[0].setup_head.remote(reporter))
        ray.get([w.initialize.remote(head_id) for w in workers])

        def train_fn(worker):
            train_args["remote_optimizer"] = worker.optimizer
            return setup_and_train(True, train_args, worker.rank)

        ray.get([w.execute.remote(train_fn) for w in workers])
        config = util.load_config(config_path, create_objects=False)
        eval_freq = config["training"]["eval_frequency"]
        print_tqdm(reporter, eval_freq)
    else:
        msg.fail(f"Strategy '{strategy}' is not implemented!", exits=1)
