from typing import Tuple, Dict
import time
from collections import defaultdict


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
        return self

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
    keys_by_node = defaultdict(list)
    for node in model.walk():
        keys = [make_key(node.id, name) for name in node.param_names]
        if keys:
            keys_by_node[node.id].extend(keys)
    key_groups = list(keys_by_node.values())
    n = max(1, len(key_groups) // num_workers)
    worker_keys = []
    start = 0
    for i in range(num_workers):
        worker_keys.append([])
        for kg in key_groups[start : start + n]:
            worker_keys[-1].extend(kg)
        start += n
    for kg in key_groups[start:]:
        worker_keys[-1].extend(kg)
    assert len(worker_keys) == num_workers, (len(worker_keys), num_workers)
    return worker_keys
