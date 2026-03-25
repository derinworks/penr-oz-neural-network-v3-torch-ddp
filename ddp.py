import os
import logging
from pathlib import Path
import sys
from multiprocessing import cpu_count
from typing import Callable
from torch import cuda, Tensor
from torch.distributed import all_reduce, get_backend, ReduceOp
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

log = logging.getLogger(__name__)

is_ddp = lambda: int(os.environ.get("RANK", -1)) != -1
ddp_rank = lambda: int(os.environ.get("RANK", 0))
ddp_local_rank = lambda: int(os.environ.get("LOCAL_RANK", 0))
ddp_world_size = lambda: int(os.environ.get("WORLD_SIZE", 1))
master_proc = lambda: (ddp_rank() == 0)

def running_on_linux() -> bool:
    # Keep Linux behavior unchanged; the issue is specific to macOS/Windows output redirection.
    return sys.platform.startswith("linux")

def launch_single_node_ddp(run_id: str, device: str, worker_op: Callable[..., None], *args):
    nproc = cuda.device_count() if device == 'cuda' else  max(1, cpu_count() // 2)
    log.info(f"Launching single node DDP run {run_id} with {nproc} processes on device {device}")
    launch_kwargs = dict(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        rdzv_backend="c10d",
        max_restarts=0,
        monitor_interval=5,
        run_id=run_id,
    )

    # Localhost rendezvous avoids noisy hostname/IPv6 lookup warnings on macOS/Windows.
    # Isolated to non-Linux because Linux doesn't exhibit this log-redirection issue.
    if not running_on_linux():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["PENR_MODEL_ID"] = run_id
        launch_kwargs["rdzv_endpoint"] = "127.0.0.1:0"

    config = LaunchConfig(**launch_kwargs)
    elastic_launch(config, entrypoint=worker_op)(*args)

def ddp_all_reduce(tensor: Tensor):
    if get_backend() == 'nccl':
        all_reduce(tensor, op=ReduceOp.AVG)
    else:
        all_reduce(tensor, op=ReduceOp.SUM)
        tensor.div_(ddp_world_size())

def reconfig_logging():
    import json
    import logging.config
    with open("log_config.json", "r") as f:
        log_config = json.load(f)

    # On non-Linux platforms, ensure worker logs are not lost due to unsupported
    # output redirection in torch.distributed.elastic.
    if is_ddp() and not running_on_linux():
        rank = ddp_rank()
        run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("PENR_MODEL_ID") or "ddp_run"
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(log_dir / f"{run_id}_rank{rank:02d}.log")
        log_config.setdefault("handlers", {})
        log_config["handlers"]["ddp_file"] = {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": file_path,
            "maxBytes": 10_485_760,
            "backupCount": 3
        }
        log_config.setdefault("root", {})
        root_handlers = log_config["root"].setdefault("handlers", [])
        if "ddp_file" not in root_handlers:
            root_handlers.append("ddp_file")

    logging.config.dictConfig(log_config)
