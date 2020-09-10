from collections import Counter


class SharedParams:
    def __init__(self):
        self._grads = {}
        self._params = {}
        self._grad_counts = Counter()
        self._transaction_ids = Counter()
        self._progress = Counter()

    def inc_progress(self, worker_id):
        self._progress[worker_id] += 1

    def get_progress(self):
        return self._progress

    def get_total_progress(self):
        return sum(self._progress.values())

    def get_transaction_id(self, key):
        return self._transaction_ids[key]

    def get_param(self, key):
        return (self._transaction_ids[key], self._params[key])

    def get_grad(self, version, key):
        if self._transaction_ids.get(key) != version:
            return None
        else:
            return self._grads.get(key)

    def get_grad_count(self, version, key):
        if self._transaction_ids.get(key) != version:
            return None
        elif key not in self._grad_counts:
            return 0
        else:
            return self._grad_counts[key]

    def set_param(self, key, value):
        self._params[key] = value
        self._transaction_ids[key] += 1
        version = self._transaction_ids[key]
        # Discard gradients when we change version.
        self._grads[key] = None
        self._grad_counts[key] = 0
        return version

    def set_grad(self, tid, key, value, grad_count):
        current_tid = self._transaction_ids.get(key)
        if tid != current_tid:
            # If we've moved past this version, discard the gradient.
            return None
        else:
            self._grads[key] = value
            self._grad_counts[key] = grad_count

    def inc_grad(self, tid, key, value, grad_count):
        current_tid = self._transaction_ids.get(key)
        if tid != current_tid:
            # If we've moved past this version, discard the gradient.
            return None
        elif key not in self._grads:
            self._grads[key] = value
            self._grad_counts[key] = 0
        else:
            self._grads[key] += value
            self._grad_counts[key] += grad_count
