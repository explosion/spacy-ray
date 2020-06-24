"""Parameter Server distributed training with Ray."""
import threading
import ray
from spacy.cli.ray_utils import create_optimizer


RAY_PS_WORKER_GPU_RESERVE = 0.9
RAY_PS_OPTIM_GPU_RESERVE = 0.1


class ParamServer:
    """A synchronous parameter server.

    Holds a Thinc optimizer internally.

    Args:
        config_path (str): Path to optimizer configuration.
        world_size (int): Size of total workers running updates.
    """
    def __init__(self, world_size):
        self.optimizer = None
        self.new_weights = None
        self.barrier = threading.Barrier(world_size)
        self.lock = threading.Lock()
        self.waiting = 0
        self.weights_dict = {}
        self.grad_dict = {}
        self.world_size = world_size

    def create_optimizer(self, config_path):
        self.optimizer = create_optimizer(config_path)

    def call(self, key, weights, gradient, *, lr_scale=1.0):
        self.lock.acquire()

        if self.waiting < self.world_size - 1:
            if self.waiting == 0:
                self.grad_dict[key] = gradient.copy()
                self.weights_dict[key] = weights.copy()
            else:
                self.grad_dict[key] += gradient
            self.waiting = self.barrier.n_waiting + 1
            self.lock.release()
            self.barrier.wait()
        else:
            self.grad_dict[key] += gradient
            self.lock.release()
            self.grad_dict[key] /= self.world_size
            new_weights, new_grads = self.optimizer(
                key,
                self.weights_dict[key],
                self.grad_dict[key],
                lr_scale=lr_scale)
            self.weights_dict[key] = new_weights
            self.grad_dict[key] = new_grads
            self.waiting = 0
            self.barrier.wait()
        return self.weights_dict[key], self.grad_dict[key]

    def fetch(self):
        return self.optimizer

    def step_schedules(self):
        self.optimizer.step_schedules()


class RayOptimizer:
    """Local handle to the parameter server."""
    local_optimizer = None

    def __init__(self, config_path, use_gpu, world_size):
        RemotePS = ray.remote(ParamServer)
        options = {"max_concurrency": world_size}
        if use_gpu >= 0:
            options["num_gpus"] = RAY_PS_OPTIM_GPU_RESERVE
        RemotePS = RemotePS.options(**options)
        self.optimizer = RemotePS.remote(config_path, world_size)
        self.optimizer.create_optimizer.remote(config_path)
        self.sync()

    def sync(self):
        self.local_optimizer = ray.get(self.optimizer.fetch.remote())

    def __call__(self, *args, **kwargs):
        weights, grads = ray.get(self.optimizer.call.remote(*args, **kwargs))
        return weights.copy(), grads.copy()

    def __getattr__(self, name):
        return getattr(self.local_optimizer, name)

    def step_schedules(self):
        self.optimizer.step_schedules.remote()
        self.sync()
