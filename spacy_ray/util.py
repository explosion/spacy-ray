import time
from typing import Tuple, Dict


KeyT = Tuple[int, str]


class Timer:
    state: str
    sum: int
    n: int

    def __init__(self, state: str):
        self.state = state
        self.sum = 0
        self.n = 0

    def __enter__(self):
        self.start = time.time()
        self.n += 1

    def __exit__(self, *args):
        interval = time.time() - self.start
        self.sum += interval


class ManyTimer:
    timers: Dict[str, Timer]

    def __init__(self):
        self.timers = {}

    def __call__(self, key: str) -> Timer:
        if key not in self.timers:
            self.timers[key] = Timer(key)
        return self.timers[key]


def set_params_proxy(model, proxy):
    """Set a 'proxy' on the internal ParamServer object for the model and
    its children. Experimental.
    """
    for node in model.walk():
        node._params.proxy = None
        for name in node.param_names:
            if node.has_param(name):
                proxy.set_param(node.id, name, node.get_param(name))
        node._params.proxy = proxy


def make_key(model_id: int, name: str) -> Tuple[int, str]:
    return (model_id, name)


def divide_params(model, num_workers):
    worker_keys = []
    for rank in range(num_workers):
        worker_keys.append([])
        for node in model.walk():
            if (node.id % num_workers) == rank:
                for param_name in node.param_names:
                    worker_keys[-1].append(make_key(node.id, param_name))
    return worker_keys
