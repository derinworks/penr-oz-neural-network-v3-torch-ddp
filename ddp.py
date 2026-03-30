import os
import logging
from pathlib import Path
import sys
from multiprocessing import cpu_count
from typing import Callable
from torch import cuda, mps, Tensor
from torch.distributed import all_reduce, get_backend, ReduceOp
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

log = logging.getLogger(__name__)

is_ddp = lambda: int(os.environ.get("RANK", -1)) != -1
ddp_rank = lambda: int(os.environ.get("RANK", 0))
ddp_local_rank = lambda: int(os.environ.get("LOCAL_RANK", 0))
ddp_world_size = lambda: int(os.environ.get("WORLD_SIZE", 1))
master_proc = lambda: (ddp_rank() == 0)

def running_on_linux() -> bool:
    return sys.platform.startswith("linux")

def detect_active_ip_family() -> str:
    import socket

    ip_family = "ipv4"
    if socket.has_ipv6:
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("::1", 1))
            s.close()
            ip_family = "ipv6"
        except Exception:
            pass

    return ip_family

def launch_single_node_ddp(run_id: str, device: str, worker_op: Callable[..., None], *args):
    if device == 'mps':
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        log.warning("MPS device detected: enabled PYTORCH_ENABLE_MPS_FALLBACK=1 "
                     "so unsupported ops (e.g. c10d::allgather_) fall back to CPU.")
    if device == 'cuda':
        nproc = cuda.device_count()
    elif device == 'mps':
        nproc = mps.device_count()
    else:
        nproc = max(1, cpu_count() // 2)
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
        use_ipv6 = detect_active_ip_family() == "ipv6"
        loopback_addr = "::1" if use_ipv6 else "127.0.0.1"
        os.environ["MASTER_ADDR"] = loopback_addr
        os.environ["GLOO_USE_IPV6"] = "1" if use_ipv6 else "0"
        launch_kwargs["local_addr"] = loopback_addr
        launch_kwargs["rdzv_endpoint"] = f"[{loopback_addr}]:0" if use_ipv6 else f"{loopback_addr}:0"
        if use_ipv6 and sys.platform == "darwin":
            os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    config = LaunchConfig(**launch_kwargs)
    elastic_launch(config, entrypoint=worker_op)(*args)

def effective_device(device: str) -> str:
    """Return the effective training device under DDP.

    The gloo backend (used for CPU and MPS) only supports CPU tensors,
    so MPS must fall back to CPU when running under DDP.
    """
    if is_ddp() and device == 'mps':
        log.warning("DDP with gloo backend does not support MPS tensors; "
                     "falling back to CPU for distributed training.")
        return 'cpu'
    return device

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
        logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
        rank = ddp_rank()
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(log_dir / f"ddp_rank{rank:02d}.log")

        log_config["handlers"]["ddp_file"] = {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": file_path,
            "maxBytes": 10_485_760,
            "backupCount": 3
        }
        root_handlers = log_config["root"]["handlers"]
        if "ddp_file" not in root_handlers:
            root_handlers.append("ddp_file")

    logging.config.dictConfig(log_config)
