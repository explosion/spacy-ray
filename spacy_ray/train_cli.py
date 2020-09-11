import time
import typer
import logging
from pathlib import Path
from spacy import util
from typing import Optional
from spacy.cli.train import import_code, parse_config_overrides
import spacy.cli._util
from thinc.api import require_gpu

from .worker import Worker, Evaluater

RAY_HELP = """Command-line interface for parallel and distributed computing via
Ray. Assumes ray is installed and that the cluster is initialized. See the
Ray documentation for details: https://ray.io.
"""
# Wrappers for Typer's annotations. Initially created to set defaults and to
# keep the names short, but not needed at the moment.
Arg = typer.Argument
Opt = typer.Option

# Create our subcommand, and install it within spaCy's CLI.
CLI = typer.Typer(name="ray", help=RAY_HELP, no_args_is_help=True)
spacy.cli._util.app.add_typer(CLI)


@CLI.command(
    "train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def ray_train_cli(
    # fmt: off
    ctx: typer.Context, # This is only used to read additional arguments
    config_path: Path = Arg(..., help="Path to config file", exists=True),
    code_path: Optional[Path] = Opt(None, "--code-path", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    strategy: str=Opt("peer_params", "--strategy", "-s", help="Which strategy to use"),
    use_gpu: int = Opt(-1, "--use-gpu", "-g", help="Use GPU"),
    num_workers: int=Opt(1, "--num_workers", "-w", help="Number of workers"),
    ray_address: str=Opt("", "--address", "-a", help="Address of ray cluster"),
    # fmt: on
):
    require_gpu(0)
    util.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    import_code(code_path)
    config = util.load_config(
        config_path, overrides=parse_config_overrides(ctx.args), interpolate=True
    )
    import ray

    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    RemoteWorker = ray.remote(Worker).options(num_gpus=int(use_gpu >= 0), num_cpus=2)
    workers = [
        RemoteWorker.remote(
            config,
            rank=rank,
            num_workers=num_workers,
            use_gpu=use_gpu,
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
