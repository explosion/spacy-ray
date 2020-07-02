"""Allreduce distributed training with Ray."""

from wasabi import msg
from spacy import util
from typing import Tuple
from thinc.types import FloatsXd

cp = None
nccl = None

try:
    import cupy as cp
    from cupy.cuda import nccl
except ImportError:
    msg.fail("Need to `pip install cupy-[driver version]` to use "
             "multi-gpu distributed training.")


def create_optimizer(config_path):
    msg.info(f"Loading config from: {config_path}")
    config = util.load_config(config_path, create_objects=False)
    util.fix_random_seed(config["training"]["seed"])
    config = util.load_config(config_path, create_objects=True)
    training = config["training"]
    return training["optimizer"]


class RayNCCLWorker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.unique_id = nccl.get_unique_id()
        self.communicator = None
        self.reporter = None

    def initialize(self, head_id, config_path):
        self.communicator = nccl.NcclCommunicator(self.world_size, head_id,
                                                  self.rank)
        self.optimizer = AllreduceOptimizer(
            config_path,  self.communicator, reporter=self.reporter)

    def setup_head(self, reporter=None):
        self.reporter = reporter
        return self.unique_id

    def execute(self, fn):
        return fn(self)


class AllreduceOptimizer:
    def __init__(self, config_path, communicator, reporter=None):
        self.optimizer = create_optimizer(config_path)
        self.communicator = communicator
        self.weights_synced = set()
        self.reporter = reporter

    def allreduce(self, tensor):
        self.communicator.allReduce(tensor.data.ptr, tensor.data.ptr,
                                    tensor.size, nccl.NCCL_FLOAT32,
                                    nccl.NCCL_SUM, cp.cuda.Stream.null.ptr)
        return tensor

    def step_schedules(self):
        if self.reporter is not None:
            self.reporter.update.remote()
        self.optimizer.step_schedules()

    def __call__(
            self,
            key: Tuple[int, str],
            weights: FloatsXd,
            gradient: FloatsXd,
            *,
            lr_scale: float = 1.0,
    ):
        if key not in self.weights_synced:
            self.weights_synced.add(key)
            weights = self.allreduce(weights) / self.communicator.size()

        gradient = self.allreduce(gradient) / self.communicator.size()
        flat_weights, gradient = self.optimizer(
            key, weights, gradient, lr_scale=lr_scale)
        return flat_weights, gradient

    def __getattr__(self, name):
        return getattr(self.optimizer, name)
