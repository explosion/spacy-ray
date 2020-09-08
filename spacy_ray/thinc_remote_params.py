from typing import Dict, Any
from timeit import default_timer as timer
from collections import defaultdict
import time
import threading
from thinc.types import FloatsXd
from .util import ManyTimer, make_key


def thread_function(next_params, ray, conn, poll):
    _last_update = 0
    while True:
        time.sleep(poll)
        updates = ray.get(conn.get_updated_params.remote(_last_update))
        new_time = timer()
        _last_update = new_time
        next_params.update(updates)


class RayProxy:
    """Proxy for the 'head' worker that owns the optimizer and pushes
    parameter updates.
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
