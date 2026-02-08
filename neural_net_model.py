import logging
import os
import platform
import multiprocessing
import random
import shutil
from contextlib import nullcontext
from typing import Tuple, Callable
import time
from datetime import datetime as dt
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
import ddp
from neural_net_layers import SoftmaxOnLast
from loaders import Loader
from mappers import Mapper


log = logging.getLogger(__name__)
MODELS_FOLDER = "models"


class NeuralNetworkModel(nn.Module):

    @staticmethod
    def _detect_shm_path() -> str:
        """Detect the best shared memory path for the current OS.

        Resolution order:
          1. ``/dev/shm`` on Linux.
          2. ``/Volumes/RAMDisk`` on macOS (if a RAM disk is mounted there).
          3. ``/tmp`` as a safe fallback.
        """
        system = platform.system()
        if system == "Linux":
            if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
                return "/dev/shm"
        elif system == "Darwin":
            if os.path.isdir("/Volumes/RAMDisk") and os.access("/Volumes/RAMDisk", os.W_OK):
                return "/Volumes/RAMDisk"
        return "/tmp"

    SHM_PATH = _detect_shm_path()

    def __init__(self, model_id: str, mapper: Mapper):
        """
        Initialize a neural network with multiple layers.
        :param mapper: maps layer creation, initialization and optimizer creation
        """
        super().__init__()
        self.model_id = model_id
        self.mapper = mapper
        self.layers = nn.ModuleList(self.mapper.to_layers())
        self._is_softmax_last = isinstance(self.layers[-1], nn.Softmax)
        self.optimizer: Optimizer = self.mapper.to_optimizer(self.parameters())
        self.progress = []
        self.avg_cost = None
        self.avg_cost_history = []
        self.stats = None
        self.status = {
            "code": "Created",
            "dt": dt.now().isoformat(),
            "message": "Model created but not yet trained."
        }

    @property
    def _weights(self) -> list[Tensor]:
        """
        :return: Weight per layer None if no weight
        """
        return [p if p.ndim == 2 else None for p in self.parameters()]

    @property
    def num_params(self) -> int:
        """
        :return: Number of model parameters
        """
        return sum([p.numel() for p in self.parameters()])

    def to(self, device: str):
        if ddp.is_ddp() and device == 'cuda':
            device = f"{device}:{ddp.ddp_local_rank()}"
            torch.cuda.set_device(device)
        super().to(device)

    @classmethod
    def get_model_path(cls, model_id):
        return os.path.join(MODELS_FOLDER, f"model_{model_id}.pth")

    def serialize(self):
        os.makedirs(MODELS_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(self.SHM_PATH, MODELS_FOLDER), exist_ok=True)
        model_path = self.get_model_path(self.model_id)
        model_data = {
            "layers": self.mapper.layers,
            "state": self.state_dict(),
            "optim": self.mapper.optimizer,
            "optim_state": self.optimizer.state_dict(),
            "progress": self.progress,
            "average_cost": self.avg_cost,
            "average_cost_history": self.avg_cost_history,
            "stats": self.stats,
            "status": self.status,
        }
        model_in_shm_path = os.path.join(self.SHM_PATH, model_path)
        if ddp.master_proc():
            log.info(f"Caching model to {model_in_shm_path}...")
        torch.save(model_data, model_in_shm_path)
        if ddp.master_proc():
            log.info(f"Model cached successfully: {model_in_shm_path}")
        p = multiprocessing.Process(target=shutil.copyfile, args=(model_in_shm_path, model_path))
        if ddp.master_proc():
            log.info(f"Offload flushing model cache {model_in_shm_path} to {model_path}...")
        p.start()

    @classmethod
    def deserialize(cls, model_id: str):
        try:
            model_path = cls.get_model_path(model_id)
            model_in_shm_path = os.path.join(cls.SHM_PATH, model_path)
            if not os.path.exists(model_in_shm_path):
                if ddp.master_proc():
                    log.info(f"Cache miss: copying from {model_path}")
                shutil.copyfile(model_path, model_in_shm_path)

            if ddp.master_proc():
                log.info(f"Retrieving model from {model_in_shm_path}...")
            data = torch.load(model_in_shm_path)
            if ddp.master_proc():
                log.info(f"Loaded model {model_id} data")
            model = cls(model_id, Mapper(data["layers"], data["optim"]))
            if ddp.master_proc():
                log.info(f"Created model {model_id}")
            model.load_state_dict(data["state"])
            if ddp.master_proc():
                log.info(f"Loaded state into model {model_id}")
            model.optimizer.load_state_dict(data["optim_state"])
            if ddp.master_proc():
                log.info(f"Loaded optimizer for model {model_id}")
            model.progress = data["progress"]
            model.avg_cost = data["average_cost"]
            model.avg_cost_history = data["average_cost_history"]
            model.stats = data["stats"]
            model.status = data["status"]
            if ddp.master_proc():
                log.info(f"Model {model_id} is fully loaded")
            return model
        except FileNotFoundError as e:
            log.error(f"File not found error occurred: {str(e)}")
            raise KeyError(f"Model {model_id} not created yet.")

    @classmethod
    def delete(cls, model_id: str):
        try:
            model_path = cls.get_model_path(model_id)
            model_in_shm_path = os.path.join(cls.SHM_PATH, model_path)
            os.remove(model_in_shm_path)
            if os.path.exists(model_path):
                os.remove(model_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    def forward(self, input_tensor: Tensor, target: Tensor=None, skip_softmax=False) -> Tuple[list[Tensor], Tensor]:
        forwarded_tensors = []
        forwarded_tensor = input_tensor
        previous_tensor = input_tensor
        layers = self.layers[:-1] if skip_softmax and self._is_softmax_last else self.layers
        for layer in layers:
            previous_tensor = forwarded_tensor
            forwarded_tensor = layer(previous_tensor)
            forwarded_tensors.append(forwarded_tensor)

        if target is None:
            cost = torch.empty(0)
        elif self._is_softmax_last:
            logits = forwarded_tensor if skip_softmax else previous_tensor
            if logits.ndim > 2 and target.ndim > 1: # e.g. transformer cost
                logits = logits.view(-1, logits.size(-1))
                target = target.view(-1)
            cost = nn.functional.cross_entropy(logits, target)
        else:
            cost = nn.functional.mse_loss(forwarded_tensor, target)

        return forwarded_tensors, cost

    @torch.no_grad()
    def compute_output(self, input_data: list, target: list | int | None = None) -> Tuple[list, float]:
        """
        Compute activated output and optionally also cost compared to the provided target vector
        without training done.
        :param input_data: Input data
        :param target: Target data (optional)
        :return: output, cost (optional)
        """
        # output is not training
        self.eval()
        self.layers.training = False
        # forward pass
        device = next(self.parameters()).device
        log.info(f"Computing model output using device {device}")
        input_tensor = torch.tensor(input_data, device=device)
        if target is not None:
            target = torch.tensor(target, device=device)
        activations, cost = self(input_tensor, target)
        # last activation  and a float cost is returned, if any
        return activations[-1].tolist(), cost.item() if cost.numel() > 0 else None

    @torch.no_grad()
    def evaluate_model(self, dataset_id: str, target_dataset_id: str | None, shard: int,
                       epochs: int, batch_size: int, block_size: int, step_size: int) -> float:
        """
        Evaluate cost compared to the provided value target without training done.
        :param dataset_id: Dataset to load
        :param target_dataset_id: Target Dataset to load (optional)
        :param shard: Dataset shard to begin from
        :param epochs: Number of evaluation iterations.
        :param batch_size: Batch size for evaluation sample.
        :param block_size: Block size (sequence length) for single evaluation sample entry
        :param step_size: Number of blocks (or sequences) to process per step
        :return: average cost
        """
        # evaluation is not training
        self.eval()
        self.layers.training = False

        # Prep evaluation data loader
        buffer_size = batch_size * block_size
        num_steps = max(1, buffer_size // (step_size * block_size * ddp.ddp_world_size()))
        begin_idx = buffer_size * ddp.ddp_rank()
        idx_offset = buffer_size * ddp.ddp_world_size()
        if ddp.master_proc():
            log.info(f"Evaluation batch size: {batch_size}, buffer size: {buffer_size}, step size: {step_size}")
            log.info(f"Evaluation calc to be done in {num_steps} step(s) repeated {epochs} epoch(s)")
        loader = Loader(dataset_id, shard, begin_idx, buffer_size, idx_offset)
        target_loader = None
        if target_dataset_id is not None:
            target_loader = Loader(target_dataset_id, shard, begin_idx, buffer_size, idx_offset)

        # determine device
        device = next(self.parameters()).device
        if ddp.master_proc():
            log.info(f"Evaluating model using device {device}")
        # evaluate cost each epoch and take average and accumulate
        avg_cost_tensor = torch.tensor(0.0).to(device)
        for epoch in range(epochs):
            # load data
            if target_loader is None:
                input_array, target_array = loader.next_batch()
            else:
                input_array = loader.next_batch(target_offset=0)
                target_array = target_loader.next_batch(target_offset=0)
            # take evaluation steps
            for step in range(num_steps):
                input_tensor = torch.tensor(input_array, dtype=torch.long).view(batch_size, block_size).to(device)
                target = torch.tensor(target_array, dtype=torch.long).view(batch_size, block_size).to(device)
                # forward pass
                _, step_cost = self(input_tensor, target, skip_softmax=True)
                # add average step cost per epoch to average cost
                avg_cost_tensor += step_cost / (epochs * num_steps)
        if ddp.is_ddp():
            # reduce avg cost among distributed workers
            ddp.ddp_all_reduce(avg_cost_tensor)
        avg_cost = avg_cost_tensor.item()
        if ddp.master_proc():
            log.info(f"Model {self.model_id}: For {epochs} evaluation(s)  Avg Cost: {avg_cost:.4f}")
        # return average sample cost
        return avg_cost

    @torch.inference_mode()
    def generate_tokens(self, input_context: list, block_size: int, max_new_tokens: int,
                        temperature=1.0, top_k: int | None=None) -> list:
        # generating is not training
        self.eval()
        self.layers.training = False
        # initialize context
        device = next(self.parameters()).device
        context = torch.tensor(input_context, dtype=torch.long, device=device)
        # log info
        top_k_msg = "" if top_k is None else f" top {top_k}"
        log.info(f"Generating at most {max_new_tokens}{top_k_msg} tokens with {temperature} temperature"
                 f" using device {device}")
        # prep Softmax layer
        softmax_layer = self.layers[-1] if self._is_softmax_last else SoftmaxOnLast
        # generate up to max new tokens
        for sample_idx in range(max_new_tokens):
            # crop context to the last block size tokens
            cropped_context = context[:, -block_size:]
            # Predict next token
            activations, _ = self(cropped_context, skip_softmax=True)
            logits: Tensor = activations[-1]
            if temperature == 0.0: # zero temperature means maximum logit is next
                next_idx = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            elif top_k is not None: # gather next from top k result by temperature ratio
                if top_k < logits.size(-1):
                    top_k_result = logits.topk(top_k, dim=-1)
                else:
                    top_k_result = logits.sort(dim=-1, descending=True)
                probs = softmax_layer.forward(top_k_result.values / temperature)
                choice = torch.multinomial(probs, num_samples=1)
                next_idx = top_k_result.indices[:, -1, :].gather(dim=1, index=choice)
            else: # next from logits by temperature ratio
                probs = softmax_layer.forward(logits / temperature)
                next_idx = torch.multinomial(probs, num_samples=1)
            # Append next token in for next prediction
            context = torch.cat((context, next_idx), dim=1)
        # extract and return tokens
        tokens = context[0].tolist()
        return tokens

    @classmethod
    def train_model_on_device(cls, model_id: str, device: str, dataset_id: str, shard: int,
                              epochs: int, batch_size: int, block_size: int, step_size: int):
        """
        Load the neural network, move to device and train starting at the specified dataset shard.
        :param model_id: Model id
        :param device: Device to move the model to
        :param dataset_id: Dataset to load
        :param shard: Dataset shard to begin from
        :param epochs: Number of training iterations.
        :param batch_size: Batch size for training sample.
        :param block_size: Block size for single training batch entry (or sequence length)
        :param step_size: Number of blocks (or sequences) to process per step
        """
        if ddp.is_ddp():
            ddp.reconfig_logging()
            backend = 'nccl' if device == 'cuda' else 'gloo'
            log.info(f"DDP local rank {ddp.ddp_local_rank()} - training model {model_id} on device {device} "
                     f"with backend {backend}")
            dist.init_process_group(backend=backend)

        model = cls.deserialize(model_id)
        model.to(device)
        if ddp.master_proc():
            log.info(f"Moved model {model_id} to device {device}")
        model.train_model(dataset_id, shard, epochs, batch_size, block_size, step_size)

        if ddp.is_ddp():
            log.info(f"Process {ddp.ddp_local_rank()} - cleaning up")
            dist.destroy_process_group()

    def train_model(self, dataset_id: str, shard: int, epochs: int, batch_size: int, block_size: int, step_size: int):
        """
        Train the neural network using the downloaded training dataset.
        :param dataset_id: Dataset to load
        :param shard: Dataset shard to begin from
        :param epochs: Number of training iterations.
        :param batch_size: Batch size for training sample.
        :param block_size: Block size for single training batch entry (or sequence length)
        :param step_size: Number of blocks (or sequences) to process per step
        """
        # Determine device
        device = next(self.parameters()).device
        if ddp.master_proc():
            log.info(f"Training model using device {device}")

        # Configure AMP autocast and gradient scaler for CUDA
        amp_ctx = nullcontext()
        amp_scaler = None
        if device.type == 'cuda':
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                amp_scaler = torch.amp.GradScaler('cuda')
            amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype)
            if ddp.master_proc():
                log.info(f"AMP enabled with dtype {amp_dtype}")

        # Prep training data loader
        buffer_size = batch_size * block_size
        num_steps = max(1, buffer_size // (step_size * block_size * ddp.ddp_world_size()))
        begin_idx = buffer_size * ddp.ddp_rank()
        idx_offset = buffer_size * ddp.ddp_world_size()
        log.info(f"Training starts from idx {begin_idx} of ds {dataset_id} at shard {shard} offset every {idx_offset}")
        if ddp.master_proc():
            log.info(f"Training batch size: {batch_size}, buffer size: {buffer_size}, step size: {step_size}")
            log.info(f"Training calc to be done in {num_steps} step(s) repeated {epochs} epoch(s)")

        loader = Loader(dataset_id,
                        begin_shard=shard,
                        begin_idx=begin_idx,
                        buffer_size=buffer_size,
                        idx_offset=idx_offset)

        # Reset model for training prep and save
        self.progress = []
        self.stats = None
        self.status = {
            "code": "Training",
            "dt": dt.now().isoformat(),
            "message": "Model is currently being trained."
        }
        last_serialized = None
        if ddp.master_proc():
            self.serialize()
            last_serialized = time.time()
        activations: list[Tensor] = []
        model = nn.parallel.DistributedDataParallel(self) if ddp.is_ddp() else self
        model.train()
        self.layers.training = True

        # Start training
        for epoch in range(epochs):
            epoch_begin_time = time.time()
            # check if training taking long
            long_training = last_serialized is not None and (epoch_begin_time - last_serialized >= 10)
            # copy weights for later update ratio calc
            prev_weights: list[Tensor] = ([None if w is None else w.clone().detach() for w in self._weights]
                                          if ddp.master_proc() else [])
            # clear gradients
            self.optimizer.zero_grad()
            # clear cost and activations
            cost = torch.tensor(0.0).to(device)
            activations.clear()
            try:
                # take training steps
                for step in range(num_steps):
                    # get next batch of training data for forward pass
                    input_tensor, target = (torch.tensor(arr, dtype=torch.long).view(batch_size, block_size).to(device)
                                            for arr in loader.next_batch())
                    # calculate step cost with optional AMP autocast
                    with amp_ctx:
                        step_activations, step_cost = model(input_tensor, target, skip_softmax=True)
                        # take average of step cost
                        avg_step_cost = step_cost / num_steps
                    # add average step to cost
                    cost += avg_step_cost.detach()
                    if ddp.master_proc():
                        # collect activations
                        activations.extend(step_activations)
                        # on last epoch or for long training intervals
                        # retain final activation gradients to collect stats
                        if epoch + 1 == epochs or long_training:
                            for a in step_activations:
                                a.retain_grad()
                    if ddp.is_ddp():
                        # turn off sync for grad accumulation steps until last step
                        model.require_backward_grad_sync = (step == num_steps - 1)
                    # back propagate to populate gradients (scaled if needed)
                    if amp_scaler is not None:
                        amp_scaler.scale(avg_step_cost).backward()
                    else:
                        avg_step_cost.backward()
            except Exception as exc:
                if ddp.master_proc():
                    log.error(f"Model {self.model_id}: Training Epoch {epoch + 1} failed: {str(exc)}")
                    self.status = {
                        "code": "Error",
                        "dt": dt.now().isoformat(),
                        "message": f"Training epoch {epoch + 1} failed: {str(exc)}"
                    }
                    self.serialize()
                raise exc
            if ddp.is_ddp():
                # reduce cost among distributed workers
                ddp.ddp_all_reduce(cost)
            # optimize parameters (with gradient unscaling if needed)
            if amp_scaler is not None:
                amp_scaler.step(self.optimizer)
                # unscale activation gradients before update changes the scale
                if ddp.master_proc():
                    inv_scale = 1.0 / amp_scaler.get_scale()
                    for a in activations:
                        if a.grad is not None:
                            a.grad.mul_(inv_scale)
                amp_scaler.update()
            else:
                self.optimizer.step()
            if device.type == 'cuda': # wait for cuda device to finish work for distributed work
                torch.cuda.synchronize()
            if ddp.master_proc():
                # Calculate training time and speed
                epoch_secs = time.time() - epoch_begin_time
                epoch_speed = buffer_size / epoch_secs
                # Record progress
                progress_cost = cost.item()
                if epoch % max(1, epochs // 100) == 0: # only 100 progress points or less stored
                    with torch.no_grad():
                        self.progress.append({
                            "dt": dt.now().isoformat(),
                            "epoch": epoch + 1,
                            "durationInSecs": epoch_secs,
                            "speedPerSec": epoch_speed,
                            "cost": progress_cost,
                            "weight_upd_ratio": [
                                None if w is None or pw is None else ((w - pw).data.std() / (w.data.std() + 1e-8)).item()
                                for pw, w in zip(prev_weights, self._weights)
                            ],
                        })
                # Log each
                log.info(f"Model {self.model_id}: Training Epoch {epoch + 1}, Cost: {progress_cost:.4f}, "
                         f"Duration: {epoch_secs:.2f} secs, Speed: {epoch_speed:.2f} tokens/sec")

            # Serialize model while long training intervals
            if ddp.master_proc() and long_training: # pragma: no cover
                self._record_training_overall_progress(activations)
                self.serialize()
                last_serialized = time.time()

        if ddp.master_proc():
            # Mark training finished
            self.status = {
                "code": "Trained",
                "dt": dt.now().isoformat(),
                "message": f"Model trained for {epochs} epochs."
            }
            # Log training is done
            log.info(f"Model {self.model_id}: Done training for {epochs} epochs.")
            # Serialize model after training
            self._record_training_overall_progress(activations)
            self.serialize()

    @torch.no_grad()
    def _record_training_overall_progress(self, activations):
        # Calculate current average progress cost
        progress_cost = [progress["cost"] for progress in self.progress]
        avg_progress_cost = sum(progress_cost) / len(self.progress)
        # Update overall average cost
        self.avg_cost = ((self.avg_cost or avg_progress_cost) + avg_progress_cost) / 2.0
        self.avg_cost_history.append(self.avg_cost)
        if len(self.avg_cost_history) > 100: #
            self.avg_cost_history.pop(random.randint(1, 98))
        # Update stats
        hist_f: Callable[[torch.return_types.histogram], Tuple[list, list]] = (
            lambda h: (h.bin_edges[:-1].tolist(), h.hist.tolist()))
        act_hist = [hist_f(torch.histogram(a.cpu(), density=True)) for a in activations]
        act_grad_hist = [([], []) if a.grad is None else hist_f(torch.histogram(a.grad.cpu(), density=True))
                         for a in activations]
        weight_grad_hist = [([], []) if w is None else hist_f(torch.histogram(w.grad.cpu(), density=True))
                            for w in self._weights]
        algos = [l.__class__.__name__.lower() for l in self.layers]
        self.stats = {
            "layers": [{
                "algo": algo,
                "activation": {
                    "mean": a.mean().item(),
                    "std": a.std().item(),
                    "saturated": (
                        (torch.norm(a, dim=-1) > 5.0) if algo == "embedding" else
                        (a.abs() > 3.0) if algo == "batchnorm1d" else
                        (a.abs() > 0.97) if algo in ["tanh", "sigmoid"] else
                        (a <= 0) if algo == "relu" else
                        (a.max(dim=-1).values > 0.97) if algo == "softmax" else
                        (a.abs() > 5.0) # if l.algo == "linear" or etc.
                    ).float().mean().item(),
                    "histogram": {"x": ahx, "y": ahy},
                },
                "gradient": {
                    "mean": a.grad.mean().item(),
                    "std": a.grad.std().item(),
                    "histogram": {"x": ghx, "y": ghy},
                } if a.grad is not None else None,
            } for algo, a, (ahx, ahy), (ghx, ghy) in zip(algos, activations, act_hist, act_grad_hist)],
            "weights": [{
                "shape": str(tuple(w.shape)),
                "data": {
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                },
                "gradient": {
                    "mean": w.grad.mean().item(),
                    "std": w.grad.std().item(),
                    "histogram": {"x": ghx, "y": ghy},
                },
            } if w is not None else None for w, (ghx, ghy) in zip(self._weights, weight_grad_hist)],
        }
        # Log training progress
        log.info(f"Model {self.model_id} - Cost: {avg_progress_cost:.4f} Overall Cost: {self.avg_cost:.4f}")
