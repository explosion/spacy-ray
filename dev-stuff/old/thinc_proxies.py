from typing import Dict, Set, Iterable, Any, Optional
from timeit import default_timer as timer
from collections import defaultdict, Counter
import time
import threading
from thinc.types import FloatsXd
from .util import ManyTimer, make_key, KeyT
from .thinc_shared_params import SharedParams


def thread_function(next_params, ray, conn, poll):
    _last_update = 0
    while True:
        time.sleep(poll)
        updates = ray.get(conn.get_updated_params.remote(_last_update))
        new_time = timer()
        _last_update = new_time
        next_params.update(updates)


class RayProxy:
    """Proxy for the workers that don't own an optimizer. Requires
    SharedOptimizer to be used.
    """

    ray: Any
    conn: Any
    _params: Dict
    _next_params: Dict
    _versions: Dict

    def __init__(self, connection, *, ray=None, use_thread=False, poll_every=0.5):
        if ray is None:
            import ray  # type: ignore
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self._poll_every = poll_every
        self._last_update = 0
        self._next_params = {}
        self._params = {}
        self._versions = {}
        self._grad_futures = defaultdict(list)
        self.timers = ManyTimer()
        self.use_thread = use_thread
        if self.use_thread:
            args = (self._next_params, self.ray, self.conn, self._poll_every)
            self.thread = threading.Thread(
                target=thread_function, args=args, daemon=True
            )

    def set_param(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a parameter to the connection."""
        key = make_key(model_id, name)
        self._params[key] = value
        self._versions[key] = self.ray.get(self.conn.set_param.remote(key, value))

    def get_param(self, model_id: int, name: str) -> FloatsXd:
        """Get a parameter from the connection."""
        key = make_key(model_id, name)
        if not self.use_thread and (timer() - self._last_update) >= self._poll_every:
            self._refresh_nexts()
        self._maybe_update_param(key)
        return self._params[key]

    def set_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Set a gradient to the connection."""
        key = make_key(model_id, name)
        self._grad_futures[key].append(
            self.conn.set_grad.remote(key, self._versions[key], value)
        )

    def inc_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
        """Increment a gradient to the connection."""
        key = make_key(model_id, name)
        self._grad_futures[key].append(
            self.conn.inc_grad.remote(key, self._versions[key], value)
        )

    def _refresh_nexts(self):
        self._await_grads()
        now_time = timer()
        self._next_params.update(
            self.ray.get(self.conn.get_updated_params.remote(self._last_update))
        )
        self._last_update = now_time

    def _await_grads(self):
        futures = []
        for g in self._grad_futures.values():
            futures.extend(g)
        self.ray.get(futures)
        self._grad_futures = defaultdict(list)

    def _maybe_update_param(self, key):
        if key in self._next_params:
            self._versions[key], self._params[key] = self._next_params.pop(key)
        if key in self._grad_futures:
            self.ray.get(self._grad_futures.pop(key))
            maybe_param = self.ray.get(
                self.conn.get_param_if_updated.remote(key, self._versions[key])
            )
            if maybe_param is not None:
                self._versions[key], self._params[key] = maybe_param
                return True
        return False

    def _sync_params(self):
        self._await_grads()
        self._params = {}
        self._versions = {}
        self._next_params = {}
        params = self.ray.get(self.conn.get_updated_params.remote(0))
        for key, (version, param) in params.items():
            self._params[key] = param
            self._versions[key] = version


class RayChildProxy:
    def __init__(self, connection, *, ray=None):
        if ray is None:
            import ray
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self._param_versions = {}

    def get_param(self, model_id: int, name: str):
        """Get a parameter from the connection."""
        key = (model_id, name)
        version, param_id = self.ray.get(self.conn.get_param.remote(model_id, name))
        self._param_versions[key] = version
        return self._decode_pointer(param_id)

    def set_param(self, model_id: int, name: str, value):
        """Child proxies don't set parameters, so this is a noop."""
        pass

    def set_grad(self, model_id: int, name: str, value):
        """Child proxies don't set gradients, so this is a noop."""
        pass

    def inc_grad(self, model_id: int, name: str, value):
        """Increment a gradient to the connection."""
        key = (model_id, name)
        version = self._param_versions[key]
        grad_count = self.ray.get(
            self.conn.get_grad_count.remote(version, model_id, name)
        )
        if grad_count is None:
            return
        elif grad_count == 0:
            self.ray.get(
                self.conn.set_grad.remote(
                    version, model_id, name, self._encode_pointer(value), 1
                )
            )
        else:
            remote_grad = self._decode_pointer(
                self.ray.get(self.conn.get_grad.remote(version, model_id, name))
            )
            if remote_grad is not None:
                value += remote_grad
            self.ray.get(
                self.conn.set_grad.remote(
                    version, model_id, name, self._encode_pointer(value), grad_count + 1
                )
            )

    def _encode_pointer(self, value):
        if value is None:
            return None
        else:
            return [self.ray.put(value)]

    def _decode_pointer(self, value):
        if value is None:
            return None
        else:
            return self.ray.get(value)[0]

    def _wait_key(self, key):
        """Await any futures for a given key."""
        self.ray.get(self._futures.get(key, []))
        self._futures[key] = []


class RayHeadProxy:
    """Proxy for the 'head' worker that owns the optimizer and pushes
    parameter updates.
    """

    def __init__(self, connection, optimizer, quorum, *, ray=None):
        if ray is None:
            import ray
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        # This 'connection' object will usually be a ray remote.
        self.conn = connection
        self.optimizer = optimizer
        self.quorum = quorum
        self._param_versions = Counter()
        self._params = {}
        self._futures = defaultdict(list)

    def get_param(self, model_id: int, name: str):
        """Get a parameter from the connection."""
        key = (model_id, name)
        return self._params[key]

    def set_param(self, model_id: int, name: str, value):
        """Set a parameter to the connection."""
        key = (model_id, name)
        self._params[key] = value
        self._param_versions[key] += 1
        version = self._param_versions[key]
        self.conn.set_param.remote(version, model_id, name, self._encode_pointer(value))

    def set_grad(self, model_id: int, name: str, value):
        """Set a gradient to the connection."""
        key = (model_id, name)
        version = self._param_versions[key]
        self.conn.set_grad.remote(
            version, model_id, name, self._encode_pointer(value), 1
        )

    def inc_grad(self, model_id: int, name: str, value):
        """Increment a gradient to the connection."""
        key = (model_id, name)
        param = self._params[key]
        version = self._param_versions[key]
        grad_count = self.ray.get(
            self.conn.get_grad_count.remote(
                version,
                model_id,
                name,
            )
        )
        if grad_count is None:
            raise ValueError("Gradient marked stale for head. Shouldn't happen?")
        else:
            if grad_count != 0:
                remote_grad = self._decode_pointer(
                    self.ray.get(
                        self.conn.get_grad.remote(
                            version,
                            model_id,
                            name,
                        )
                    )
                )
                if remote_grad is not None:
                    value += remote_grad

            if (grad_count + 1) >= self.quorum:
                param, _ = self.optimizer(key, param, value)
                self._params[key] = param
                self._param_versions[key] = version + 1
                self.conn.set_param.remote(
                    version + 1, model_id, name, self._encode_pointer(param)
                )
            else:
                self.conn.set_grad.remote(
                    version, model_id, name, self._encode_pointer(value), grad_count + 1
                )

    def step_schedules(self):
        self.optimizer.step_schedules()

    def _encode_pointer(self, value):
        return [self.ray.put(value)]

    def _decode_pointer(self, value):
        if value is None:
            return None
        else:
            return self.ray.get(value)[0]


class RayPeerProxy:
    """Proxy for workers where each worker owns some of the parameters. For
    parameters they don't own, workers will pull parameters and push gradients.
    For parameters they do own, they pull gradients, make the update, and push
    parameters.
    """

    ray: Any
    _params: Dict[KeyT, FloatsXd]
    _versions: Dict[KeyT, int]
    _owned_keys: Set[KeyT]
    _grads: Dict

    def __init__(
        self,
        peers: Dict[KeyT, "Remote"],
        optimizer,
        keys: Iterable[KeyT],
        *,
        grads_per_update: int = 2,
        ray=None
    ):
        if ray is None:
            import ray  # type: ignore
        # Pass in 'ray' so that we can test with a mock object.
        self.ray = ray
        self.optimizer = optimizer
        self.grads_per_update = grads_per_update
        self.peers = dict(peers)
        self._owned_keys = set(keys)
        self.other_workers = set()
        for key, peer in self.peers.items():
            if key not in self._owned_keys and peer not in self.other_workers:
                self.other_workers.add(peer)
        self._params = {}
        self._versions = Counter()
        self._grads = {}
        self._grad_counts = Counter()
        self._futures = []

    def check_version(self, key: KeyT, version: int) -> Optional[bool]:
        if key not in self._versions:
            return None
        elif self._versions[key] != version:
            return False
        else:
            return True

    def set_param(self, id, name, value: FloatsXd) -> None:
        """Set a parameter to the connection."""
        key = make_key(id, name)
        if key in self._owned_keys or key not in self._params:
            self._params[key] = value
            self._versions[key] += 1
            self._grads[key] = None
            self._grad_counts[key] = 0

    def send_param(self, key):
        param = self._params[key]
        version = self._versions[key]
        for peer in self.other_workers:
            peer.set_param.remote(key, version, param)

    def receive_param(self, key, version, value: FloatsXd) -> None:
        """Let the connection push a parameter to us."""
        self._params[key] = value
        self._versions[key] = version
        self._grads[key] = None
        self._grad_counts[key] = 0

    def get_param(self, id, name) -> FloatsXd:
        key = make_key(id, name)
        self._maybe_update_param(key)
        return self._params[key]

    def set_grad(self, id, name, value: FloatsXd) -> None:
        """Set a gradient to the connection."""
        key = make_key(id, name)
        if key in self._owned_keys:
            self._grads[key] = value
            self._grad_counts[key] = 1

    def inc_grad(self, id, name, value: FloatsXd) -> None:
        """Increment a gradient to the connection."""
        key = make_key(id, name)
        self._grad_counts[key] += 1
        if key not in self._owned_keys:
            peer = self.peers[key]
            peer.inc_grad.remote(key, self._versions[key], value)
        else:
            if self._grads.get(key) is None:
                self._grads[key] = value.copy()
            else:
                self._grads[key] += value

    def _maybe_update_param(self, key):
        if key not in self._owned_keys:
            return False
        elif self._grad_counts[key] < self.grads_per_update:
            return False
        else:
            self._versions[key] += 1
            param, _ = self.optimizer(key, self._params[key], self._grads[key])
            self._params[key] = param
            self._grads[key] = None
            self._grad_counts[key] = 0
            self.send_param(key)
            return True
