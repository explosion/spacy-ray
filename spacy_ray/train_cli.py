from typing import Optional
import time
import typer
import logging
from pathlib import Path
from spacy import util
from spacy.cli._util import import_code, parse_config_overrides, Arg, Opt, app
from thinc.api import require_gpu, Config

from .worker import Worker, Evaluater


RAY_HELP = """Command-line interface for parallel and distributed computing via
Ray. Assumes Ray is installed and that the cluster is initialized. See the
Ray documentation for details: https://ray.io.
"""

# Create our subcommand, and install it within spaCy's CLI
ray_cli = typer.Typer(name="ray", help=RAY_HELP, no_args_is_help=True)
app.add_typer(ray_cli)


@ray_cli.command(
    "train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def ray_train_cli(
    # fmt: off
    ctx: typer.Context,  # This is only used to read additional arguments
    config_path: Path = Arg(..., help="Path to config file", exists=True),
    code_path: Optional[Path] = Opt(None, "--code-path", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    strategy: str = Opt("peer_params", "--strategy", "-s", help="Which strategy to use"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    num_workers: int = Opt(1, "--n-workers", "-w", help="Number of workers"),
    ray_address: str = Opt("", "--address", "-a", help="Address of ray cluster"),
    # fmt: on
):
    """
    Train a spaCy model using Ray for parallel training.
    """
    require_gpu(0)
    util.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    import_code(code_path)
    config_overrides = parse_config_overrides(ctx.args)
    config = util.load_config(config_path, overrides=config_overrides, interpolate=True)
    ray_train(config, ray_address=ray_address, num_workers=num_workers, use_gpu=use_gpu)


def ray_train(
    config: Config, *, ray_address: str = "", num_workers: int = 1, use_gpu: int = -1
) -> None:
    # We're importing Ray here so it doesn't need to be imported when spaCy /
    # spaCy's CLI is imported (which would otherwise take too long)
    import ray

    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init(ignore_reinit_error=True)
    RemoteWorker = ray.remote(Worker).options(num_gpus=int(use_gpu >= 0), num_cpus=2)
    workers = [
        RemoteWorker.remote(
            config, rank=rank, num_workers=num_workers, use_gpu=use_gpu,
        )
        for rank in range(num_workers)
    ]
    for worker in workers:
        ray.get(worker.set_proxy.remote(workers))
    evaluater = ray.remote(Evaluater).remote()
    for worker in workers:
        ray.get(worker.train.remote(workers, evaluater))
    todo = list(workers)
    while todo:
        time.sleep(1)
        todo = [w for w in workers if ray.get(w.is_running.remote())]
