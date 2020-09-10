from typing import Dict, Tuple, Any, Optional, cast
from dataclasses import dataclass
from collections import Counter
import threading
from timeit import default_timer as timer
from thinc.types import FloatsXd
from thinc.api import Config, Optimizer, registry
from .util import KeyT


@dataclass
class ParamData:
    key: Tuple[int, str]
    version: int
    timestamp: Any
    value: FloatsXd
    grads: Optional[FloatsXd]
    grad_count: int


class SharedOptimizer:
    """Provide access to an optimizer for multiple workers. Designed to be
    used as a ray remote actor, connected to a ParamServer via RayProxy.
    """

    ray: Any
    quorum: int
    optimizer: Optimizer
    write_lock: threading.Lock
    _params: Dict[KeyT, ParamData]
    _progress: Counter
    _n_kept: int
    _n_dropped: int
    _n_updates: int

    def __init__(self, optimizer_config: Config, quorum: int, ray=None):
        if ray is None:
            import ray  # type: ignore
        self.ray = ray
        self.quorum = quorum
        self.optimizer = registry.make_from_config(optimizer_config)["optimizer"]
        self._params = {}
        self._progress = Counter()
        self._n_kept = 0
        self._n_dropped = 0
        self._n_updates = 0
        self.write_lock = threading.Lock()

    def get_quorum(self) -> int:
        return self.quorum

    def inc_progress(self, worker_id: int) -> None:
        self._progress[worker_id] += 1

    def get_progress(self) -> Counter:
        return self._progress

    def get_total_progress(self) -> float:
        return sum(self._progress.values())

    def step_schedules(self) -> None:
        self.optimizer.step_schedules()

    def get_transaction_id(self, key: KeyT) -> int:
        return self._params[key].version

    def get_param(self, key: KeyT) -> Tuple[int, FloatsXd]:
        return (self._params[key].version, self._params[key].value)

    def get_percent_dropped(self) -> float:
        total = self._n_dropped + self._n_kept
        if total == 0:
            return total
        else:
            return self._n_dropped / total

    def get_param_if_updated(
        self, key: KeyT, version: int
    ) -> Optional[Tuple[int, FloatsXd]]:
        if key not in self._params:
            raise KeyError("wat")
        elif self._params[key].version == version:
            return None
        else:
            return (self._params[key].version, self._params[key].value)

    def get_updated_params(self, since: float) -> Dict:
        """Return a dict with params that have changed since a given timestamp."""
        updates = {}
        for key, p in self._params.items():
            if p.timestamp >= since:
                updates[key] = (p.version, p.value)
        return updates

    def set_param(self, key: KeyT, value: FloatsXd) -> int:
        with self.write_lock:
            if key in self._params:
                version = self._params[key].version + 1
            else:
                version = 0
            self._params[key] = ParamData(
                key=key,
                value=value,
                version=version,
                grads=None,
                grad_count=0,
                timestamp=timer(),
            )
            return self._params[key].version

    def set_grad(self, tid: int, key: KeyT, value: FloatsXd) -> None:
        with self.write_lock:
            if key not in self._params:
                return None
            elif tid != self._params[key].version:
                # If we've moved past this version, discard the gradient.
                return None
            else:
                self._params[key].grads = value.copy()
                self._params[key].grad_count = 1
                self._update_if_quorum(key)

    def inc_grad(self, key: KeyT, tid: int, value: FloatsXd) -> None:
        with self.write_lock:
            if key not in self._params:
                return None
            elif tid != self._params[key].version:
                self._n_dropped += 1
                return None
            elif self._params[key].grads is None:
                self._n_kept += 1
                self._params[key].grads = value.copy()
                self._params[key].grad_count = 1
                self._update_if_quorum(key)
            else:
                self._n_kept += 1
                self._params[key].grads += value
                self._params[key].grad_count += 1
                self._update_if_quorum(key)

    def _update_if_quorum(self, key: KeyT) -> None:
        if key not in self._params:
            return
        if self._params[key].grad_count >= self.quorum:
            grads = cast(FloatsXd, self._params[key].grads)
            # The optimizer call changes internal state, so we need to worry
            # about concurrency on it.
            params, _ = self.optimizer(key, self._params[key].value.copy(), grads)
            new_param = ParamData(
                key=key,
                value=params,
                version=self._params[key].version + 1,
                grads=None,
                grad_count=0,
                timestamp=timer(),
            )
            self._params[key] = new_param
            self._n_updates += 1
