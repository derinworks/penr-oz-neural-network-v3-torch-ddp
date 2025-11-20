# penr-oz-neural-network-v3-torch-ddp
Version 3 Implementation of Neural Network service leveraging pytorch distributed data platform (ddp)

This repository demonstrates same key concepts in neural networks as in [penr-oz-neural-network-v2-torch-nn](https://github.com/derinworks/penr-oz-neural-network-v2-torch-nn) 
with automatic gradient descent calculations relying on PyTorch library leveraging Neural Network (nn) package, 
Distributed Data Parallel (ddp) feature to support scaling to multiple GPU (CUDA) devices, and changes to API to support
downloading/sharding training data for local read instead of an API payload.

## What this service provides

* **FastAPI interface for model lifecycle management** — create, train, evaluate, generate, and delete neural network models over HTTP. Endpoints are defined in `main.py` and are exposed both through Swagger (`/docs`) and a lightweight dashboard (`/dashboard`).
* **Dataset tokenization and sharding** — datasets are downloaded asynchronously, tokenized, and sharded locally before training to reduce payload size.
* **Single-node Distributed Data Parallel (DDP)** — the `ddp.launch_single_node_ddp` helper spins up one process per available CUDA device (or a CPU-backed fallback) to accelerate training without additional orchestration.
* **Transparent logging and diagnostics** — `log_config.json` drives application logging, while `/progress` and `/stats` report training status and metrics during long-running jobs.

### Core components

* `main.py` — FastAPI app with routes for dataset management, tokenization, model CRUD, training, evaluation, and text generation.
* `ddp.py` — helper to launch single-node elastic DDP jobs, normalize all-reduce behavior across backends, and reconfigure logging in worker processes.
* `neural_net_model.py` — orchestrates model serialization/deserialization, training on a target device, and metric tracking for `/progress` and `/stats` responses.
* `templates/dashboard.html` & `static` assets — provide a simple monitoring view for training progress and metrics.

Implementation follows:
* [nn-zero-to-hero lectures](https://github.com/karpathy/nn-zero-to-hero)
* [makemore](https://github.com/karpathy/makemore)
* [build-nanogpt](https://github.com/karpathy/build-nanogpt)
* [nanoGPT](https://github.com/karpathy/nanoGPT)

### Backpropagation: Auto Gradient Calculation

The gradients are automatically computed using [PyTorch](https://github.com/pytorch/pytorch) and 
the [PyTorch Neural Network package](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial)

### Scaling Calculation Speed and Concurrency

This is done by leveraging [PyTorch Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## Quickstart Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/derinworks/penr-oz-neural-network-v3-torch-ddp.git
   cd penr-oz-neural-network-v3-torch-ddp
   ```

2. **Create and Activate a Virtual Environment**:
   - **Install [python 3.10](https://www.python.org/downloads/release/python-31018/)**
     ```bash
     $ python3 --version
     Python 3.10.18
     ```
   - **Create**:
     ```bash
     python3 -m venv venv
     ```
   - **Activate**:
     - On Unix or macOS:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Service**:
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --log-config log_config.json
   ```

5. **Interact with the Service**
Test the endpoints using Swagger at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

6. **Interact with the Dashboard**
Diagnose model training at [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard).

### Typical API workflow

The API is asynchronous where long-running work is involved (dataset downloads and training return `202 Accepted`). A minimal end-to-end flow:

1. **Download and shard a dataset**
   ```bash
   curl -X POST http://127.0.0.1:8000/dataset/ \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "tiny-shakespeare",
       "encoding": "gpt2",
       "path": "andriotis/tiny-shakespeare-karpathy",
       "name": "default",
       "split": "train",
       "shard_size": 100000
     }'
   ```

2. **Create a model definition**
   ```bash
   curl -X POST http://127.0.0.1:8000/model/ \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "gpt-example",
       "layers": [...],
       "optimizer": {"adamw": {"lr": 0.0006, "betas": [0.9, 0.95], "eps": 1e-8}}
     }'
   ```

3. **Launch training (DDP will spawn one worker per device)**
   ```bash
   curl -X PUT http://127.0.0.1:8000/train/ \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "gpt-example",
       "dataset_id": "tiny-shakespeare",
       "device": "cuda",
       "shard": 0,
       "epochs": 4,
       "batch_size": 2,
       "block_size": 1024
     }'
   ```

4. **Monitor progress and metrics**
   ```bash
   curl "http://127.0.0.1:8000/progress/?model_id=gpt-example"
   curl "http://127.0.0.1:8000/stats/?model_id=gpt-example"
   ```

5. **Generate or decode text**
   ```bash
   curl -X POST http://127.0.0.1:8000/generate/ \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "gpt-example",
       "input": [[0]],
       "block_size": 1024,
       "max_new_tokens": 20,
       "temperature": 1.0,
       "top_k": 10
     }'
   ```

### Distributed training notes

* Set `device` to `cuda` to automatically launch one process per available GPU. On CPU-only environments the launcher falls back to half the available cores to avoid oversubscription.
* The launcher uses PyTorch Elastic under the hood, so ensure `torch.distributed` is available in your environment. No external rendezvous service is required for single-node jobs.
* Training and dataset downloads are guarded by per-model/per-dataset locks to prevent conflicting work; a `409 Conflict` response indicates that a job is already in progress.

---

## Testing and Coverage

To ensure code quality and maintainability, follow these steps to run tests and check code coverage:

1. **Run Tests with Coverage**:
   Execute the following commands to run tests and generate a coverage report:
   ```bash
   coverage run -m pytest
   coverage report
   ```

2. **Generate HTML Coverage Report** (Optional):
   For a detailed coverage report in HTML format:
   ```bash
   coverage html
   ```
   Open the `htmlcov/index.html` file in a web browser to view the report.
