import spacy
from spacy.language import Language

from . import mock_ray
from ..worker import Worker

# I don't normally go for so much mocking; I usually think it's pointless to
# test something so different. But this time it seems worth trying, just to
# get some basic tests, which would otherwise be hard.

# These tests currently really do nothing, I'm just laying out a skeleton while
# I think of what could be meaningfully tested this way. Maybe I'll delete it
# all in the end.


class TestWorker(Worker):
    def _load_nlp_and_config(self, config):
        return None, None

    def _initialize_models(self, nlp, config):
        return None, None


def test_worker_init():
    # Get a blank valid config
    nlp = spacy.blank("en")
    nlp.config["paths"]["train"] = ""
    nlp.config["paths"]["dev"] = ""
    worker = Worker(nlp.config, rank=1, num_workers=2, use_gpu=-1, ray=mock_ray)
    assert isinstance(worker.nlp, Language)
