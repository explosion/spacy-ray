from typing import Optional
import time
import typer
import logging
from pathlib import Path
from spacy.util import load_config, logger
from spacy.cli._util import parse_config_overrides, Arg, Opt, app
from spacy.cli._util import setup_gpu, show_validation_error
from thinc.api import Config

from .worker import Worker, Evaluator


RAY_HELP = """CLI for parallel and distributed computing via
Ray. See the Ray documentation for details: https://ray.io.
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
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    output_path: Optional[Path] = Opt(None, "--output", "--output-path", "-o", help="Output directory or remote storage URL for saving trained pipeline"),
    num_workers: int = Opt(1, "--n-workers", "-w", help="Number of workers"),
    ray_address: Optional[str] = Opt(None, "--address", "-a", help="Address of ray cluster"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    # fmt: on
):
    """
    Train a spaCy pipeline using Ray for parallel training.
    """
    # TODO: wire up output path
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    setup_gpu(use_gpu)
    overrides = parse_config_overrides(ctx.args)
    with show_validation_error(config_path):
        config = load_config(config_path, overrides=overrides, interpolate=False)
    ray_train(
        config,
        ray_address=ray_address,
        num_workers=num_workers,
        use_gpu=use_gpu,
        code_path=code_path,
    )


def ray_train(
    config: Config,
    *,
    ray_address: Optional[str] = None,
    num_workers: int = 1,
    use_gpu: int = -1,
    code_path: Optional[Path] = None,
) -> None:
    # We're importing Ray here so it doesn't need to be imported when spaCy /
    # spaCy's CLI is imported (which would otherwise take too long)
    import ray

    if ray_address is not None:
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
            code_path=code_path,
        )
        for rank in range(num_workers)
    ]
    for worker in workers:
        ray.get(worker.set_proxy.remote(workers))
    evaluator = ray.remote(Evaluator).remote()
    for worker in workers:
        ray.get(worker.train.remote(workers, evaluator))
    todo = list(workers)
    while todo:
        time.sleep(1)
        todo = [w for w in workers if ray.get(w.is_running.remote())]
