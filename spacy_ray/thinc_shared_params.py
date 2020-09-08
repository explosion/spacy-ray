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

    def get_param(self, model_id, name):
        key = (model_id, name)
        return (self._transaction_ids[key], self._params[key])

    def get_grad(self, version, model_id, name):
        key = (model_id, name)
        if self._transaction_ids.get(key) != version:
            return None
        else:
            return self._grads.get(key)

    def get_grad_count(self, version, model_id, name):
        key = (model_id, name)
        if self._transaction_ids.get(key) != version:
            return None
        elif key not in self._grad_counts:
            return 0
        else:
            return self._grad_counts[key]

    def set_param(self, version, model_id, name, value):
        key = (model_id, name)
        self._params[key] = value
        self._transaction_ids[key] = version
        # Discard gradients when we change version.
        self._grads[key] = None
        self._grad_counts[key] = 0

    def set_grad(self, tid, model_id, name, value, grad_count):
        key = (model_id, name)
        current_tid = self._transaction_ids.get(key)
        if tid != current_tid:
            # If we've moved past this version, discard the gradient.
            return None
        else:
            self._grads[key] = value
            self._grad_counts[key] = grad_count
