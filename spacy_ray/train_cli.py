from spacy.cli._app import app, Arg, Opt
import ray
import time
import os

from .worker import Worker, Evaluater 


@app.command("ray-train")
def ray_train_cli(
    # fmt: off
    config_path: Path = Arg(..., help="Path to config file", exists=True),
    code_path: Optional[Path] = Opt(None, "--code-path", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    use_gpu: int = Opt(-1, "--use-gpu", "-g", help="Use GPU"),
    num_workers: int=Opt(1, "--num_workers", "-w", help="Number of workers"),
    ray_address: str=Opt("", "--address", "-a", help="Address of ray cluster")
    # fmt: on
):
    util.logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    verify_cli_args(config_path, output_path)
    import_code(code_path)
    config = load_config(
        config_path,
        overrides=parse_config_overrides(ctx.args),
        interpolate=True
    )
    distributed_setup_and_train(config, ray_address, use_gpu, num_workers)


def distributed_setup_and_train(config, use_gpu, num_workers, ray_address):
    if ray_address is not None:
        ray.init(address=ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    RemoteWorker = ray.remote(Worker).options(
        num_gpus=int(use_gpu >= 0),
        num_cpus=2
    )
    workers = [
        RemoteWorker.remote(
            rank,
            num_workers,
            use_gpu,
            config_path,
            config_overrides
        )
        for rank in range(num_workers)
    ]
    evaluater = ray.remote(Evaluater).remote()
    conn = ray.remote(SharedParams).options(num_gpus=0).remote()
    futures = []
    for i, w in enumerate(workers):
        futures.append(w.train.remote(use_gpu, conn, evaluater))
    ray.get(futures)
