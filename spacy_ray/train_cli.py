import typer
import logging
from pathlib import Path
from spacy import util
from typing import Optional
from spacy.cli.train import import_code, parse_config_overrides
import spacy.cli._util

from .worker import Worker, Evaluater
from .thinc_remote_params import SharedOptimizer

RAY_HELP = f"""Command-line interface for parallel and distributed computing via
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
    "train",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True
    }
)
def ray_train_cli(
    # fmt: off
    ctx: typer.Context, # This is only used to read additional arguments
    config_path: Path = Arg(..., help="Path to config file", exists=True),
    code_path: Optional[Path] = Opt(None, "--code-path", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    use_gpu: int = Opt(-1, "--use-gpu", "-g", help="Use GPU"),
    num_workers: int=Opt(1, "--num_workers", "-w", help="Number of workers"),
    ray_address: str=Opt("", "--address", "-a", help="Address of ray cluster")
    # fmt: on
):
    util.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    import_code(code_path)
    config = util.load_config(
        config_path, overrides=parse_config_overrides(ctx.args), interpolate=True
    )
    distributed_setup_and_train(config, use_gpu, num_workers, ray_address)


def distributed_setup_and_train(config, use_gpu, num_workers, ray_address, ray=None):
    if ray is None:
        import ray
    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    RemoteWorker = ray.remote(Worker).options(num_gpus=int(use_gpu >= 0), num_cpus=2)
    workers = [
        RemoteWorker.remote(rank, num_workers, use_gpu, config)
        for rank in range(num_workers)
    ]
    evaluater = ray.remote(Evaluater).remote()
    conn = (
        ray.remote(SharedOptimizer)
        .options(num_gpus=0)
        .remote(
            ray.get(workers[0].get_optimizer.remote()),
            ray.get(workers[0].get_quorum.remote()),
        )
    )
    futures = []
    for i, w in enumerate(workers):
        futures.append(w.train.remote(use_gpu, conn, evaluater))
    ray.get(futures)
