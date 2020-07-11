import pytest
import numpy as np

import ray
from spacy_ray.param_server import ParamServer, RayOptimizer
from spacy_ray import distributed_setup_and_train


@pytest.fixture
def ray_start_3_cpus():
    address_info = ray.init(num_cpus=3)
    yield address_info
    ray.shutdown()


@pytest.fixture
def ray_start_2_gpus():
    address_info = ray.init(num_gpus=2)
    yield address_info
    ray.shutdown()



def test_10_step(ray_start_3_cpus):  # noqa: F811
    class ToyServer(ParamServer):
        def create_optimizer(self, config_path):
            self.optimizer = lambda key, weights, grad, lr_scale: (
                np.zeros(1), np.zeros(1))

    RemoteServer = ray.remote(ToyServer).options(max_concurrency=2)

    server = RemoteServer.remote(world_size=2)
    ray.get(server.create_optimizer.remote(None))

    for i in range(10):
        result_id = server.call.remote(1, np.zeros(1), np.zeros(1))
        finished, waiting = ray.wait([result_id], timeout=2)
        assert not finished

        result_id = server.call.remote(1, np.zeros(1), np.zeros(1))
        finished, waiting = ray.wait([result_id], timeout=2)
        assert finished
        ray.get(finished)


def test_full_path(ray_start_3_cpus):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(TEST_CONFIG_STR)
    optimizer = RayOptimizer(tmpfile.name, use_gpu=False, world_size=1)

    config = util.load_config(tmpfile.name, create_objects=True)
    training = config["training"]
    msg.info("Creating nlp from config")
    nlp_config = config["nlp"]
    nlp = util.load_model_from_config(nlp_config)
    for name, proc in nlp.pipeline:
        if hasattr(proc, "model"):
            proc.model.finish_update(optimizer)


def test_gpu_allreduce(ray_start_2_gpus):
    distributed_setup_and_train(use_gpu, num_workers=2, strategy="allreduce")

