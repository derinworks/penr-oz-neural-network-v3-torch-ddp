import os
import logging
from multiprocessing import cpu_count
from typing import Callable
from torch import cuda
import torch.distributed as dist
from torch.distributed.launcher.api import elastic_launch, LaunchConfig

log = logging.getLogger(__name__)

is_ddp = lambda: dist.is_available() and dist.is_initialized()
ddp_rank = lambda: dist.get_rank() if is_ddp() else 0
ddp_local_rank = lambda: int(os.environ.get("LOCAL_RANK", 0))
ddp_world_size = lambda: dist.get_world_size() if is_ddp() else 1
master_proc = lambda: (ddp_rank() == 0)

def launch_single_node_ddp(run_id: str, device: str, worker_op: Callable[..., None], *args):
    nproc = cuda.device_count() if device == 'cuda' else  max(1, cpu_count() // 2)
    log.info(f"Launching single node DDP run {run_id} with {nproc} processes on device {device}")
    config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=nproc, rdzv_backend="c10d",
                          max_restarts=0, monitor_interval=5, run_id=run_id)
    elastic_launch(config, entrypoint=worker_op)(*args)

def reconfig_logging():
    import json
    import logging.config
    with open("log_config.json", "r") as f:
        log_config = json.load(f)
    logging.config.dictConfig(log_config)
