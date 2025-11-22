# Neural Network Service v3 - PyTorch DDP

A FastAPI-based neural network training and inference service with PyTorch Distributed Data Parallel (DDP) support for multi-GPU training.

## Overview

This service provides a REST API for creating, training, evaluating, and generating text with neural network models (primarily GPT-style transformers). It leverages PyTorch's automatic differentiation and Distributed Data Parallel capabilities to scale training across multiple GPU devices.

### Key Features

- **Automatic Gradient Calculation**: Uses PyTorch's autograd system
- **Distributed Training**: PyTorch DDP for multi-GPU/CPU parallel training
- **REST API**: Full-featured FastAPI service with Swagger docs
- **Web Dashboard**: Real-time training diagnostics at `/dashboard`
- **Dataset Management**: HuggingFace dataset integration with local sharding
- **Tokenization**: TikToken encoding support (e.g., GPT-2)
- **Model Persistence**: Serialization with shared memory caching (`/dev/shm`) for performance

### Architecture

**Core Components:**
- `main.py` - FastAPI application with endpoints for model/dataset operations
- `neural_net_model.py` - Neural network model implementation (PyTorch `nn.Module`)
- `neural_net_layers.py` - Custom layers (CausalSelfAttention, Residual, etc.)
- `ddp.py` - Distributed Data Parallel utilities and launcher
- `loaders.py` - Dataset downloading and loading from HuggingFace
- `mappers.py` - Maps layer/optimizer configurations to PyTorch objects
- `tokenizers.py` - TikToken tokenizer wrapper

**Inspired By:**
- Andrej Karpathy's [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- [makemore](https://github.com/karpathy/makemore)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## Setup

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (optional, for GPU training)

### Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- FastAPI + Uvicorn (web framework)
- PyTorch (neural network library)
- TikToken (tokenization)
- HuggingFace datasets (data loading)
- Pytest + Coverage (testing)

## Running the Service

```bash
# Start the service
python main.py

# Or with uvicorn directly
uvicorn main:app --log-config log_config.json
```

**Access Points:**
- API Docs: http://127.0.0.1:8000/docs
- Dashboard: http://127.0.0.1:8000/dashboard
- Root: http://127.0.0.1:8000/ (redirects to dashboard)

## Common Workflows

### 1. Download and Prepare Dataset

```bash
# POST /dataset/
curl -X POST http://127.0.0.1:8000/dataset/ \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "tiny-shakespeare",
    "path": "andriotis/tiny-shakespeare-karpathy",
    "name": "default",
    "split": "train",
    "shard_size": 100000,
    "encoding": "gpt2"
  }'
```

### 2. Create a Model

```bash
# POST /model/
# Example: Simple GPT-style transformer with 12 layers
curl -X POST http://127.0.0.1:8000/model/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-small",
    "layers": [...],  # See API docs for layer configuration
    "optimizer": {"adamw": {"lr": 6e-4, "betas": [0.9, 0.95]}}
  }'
```

### 3. Train the Model

```bash
# PUT /train/
curl -X PUT http://127.0.0.1:8000/train/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-small",
    "device": "cuda",
    "dataset_id": "tiny-shakespeare",
    "shard": 1,
    "epochs": 4,
    "batch_size": 2,
    "block_size": 1024
  }'
```

### 4. Monitor Training Progress

```bash
# GET /progress/?model_id=gpt-small
curl http://127.0.0.1:8000/progress/?model_id=gpt-small
```

### 5. Generate Text

```bash
# First tokenize input
curl -X POST http://127.0.0.1:8000/tokenize/ \
  -H "Content-Type: application/json" \
  -d '{"encoding": "gpt2", "text": "To be or not to be"}'

# Then generate
curl -X POST http://127.0.0.1:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-small",
    "input": [[1234, 5678]],
    "block_size": 1024,
    "max_new_tokens": 100,
    "temperature": 1.0
  }'

# Decode generated tokens
curl -X POST http://127.0.0.1:8000/decode/ \
  -H "Content-Type: application/json" \
  -d '{"encoding": "gpt2", "tokens": [...]}'
```

## API Endpoints

### Model Operations
- `POST /model/` - Create a new model
- `GET /stats/?model_id=...` - Get model statistics
- `GET /progress/?model_id=...` - Get training progress
- `DELETE /model/?model_id=...` - Delete a model

### Dataset Operations
- `POST /dataset/` - Download dataset from HuggingFace
- `GET /dataset/?dataset_id=...` - List dataset files
- `DELETE /dataset/?dataset_id=...` - Delete a dataset

### Training & Evaluation
- `PUT /train/` - Train model (asynchronous, uses DDP)
- `POST /evaluate/` - Evaluate model on dataset
- `POST /output/` - Compute model output for given input

### Text Generation
- `POST /tokenize/` - Tokenize text
- `POST /generate/` - Generate tokens from model
- `POST /decode/` - Decode tokens to text

## Testing

```bash
# Run all tests
python -m pytest -v

# Run tests with coverage
coverage run -m pytest
coverage report

# Generate HTML coverage report
coverage html
# Open htmlcov/index.html in browser
```

**Test Files:**
- `test_main.py` - API endpoint tests (29 tests)
- `test_neural_net_model.py` - Model implementation tests (43 tests)
- `test_neural_net_layers.py` - Custom layer tests (12 tests)
- `test_loaders.py` - Dataset loader/downloader tests (7 tests)
- `test_mappers.py` - Layer/optimizer mapper tests (3 tests)
- `test_tokenizers.py` - Tokenization tests (4 tests)
- `test_ddp.py` - Distributed training tests (16 tests)

**Platform-Specific Tests:**

Some tests require Linux-specific features (e.g., `/dev/shm` for shared memory caching) and will be automatically skipped on macOS/Windows:
- `test_train_*` - Full training integration tests with model persistence
- `test_cache_miss` - Shared memory cache behavior
- `test_delete` - Model deletion with shared memory cleanup

These tests will run automatically on Linux systems where `/dev/shm` is available.

## Distributed Training

The service automatically detects available GPUs and distributes training:

- **CUDA devices**: Uses all available GPUs via NCCL backend
- **CPU only**: Uses half of available CPU cores via Gloo backend

DDP is launched automatically when training via `PUT /train/` endpoint. The training runs in a background process, allowing async operation.

## Model Persistence

Models are saved with:
- Layer configuration
- Model weights (state_dict)
- Optimizer state
- Training progress and statistics

**Storage Strategy:**
1. Fast cache in `/dev/shm` (shared memory) for quick access
2. Persistent storage in `models/` directory
3. Asynchronous flush from cache to disk

## Custom Layers

### CausalSelfAttention
Multi-head causal self-attention for transformer models

### PositionEmbedding
Automatic position encoding for sequence data

### Summation
Sums outputs of multiple parallel layers (e.g., token + position embeddings)

### ResidualConnection
Residual connections for deeper networks

### SoftmaxOnLast
Applies softmax only to the last token (for next-token prediction)

## Configuration

### Layer Configuration
Layers are configured via JSON with PyTorch nn module names:
```json
{
  "embedding": {"num_embeddings": 50304, "embedding_dim": 768},
  "layernorm": {"normalized_shape": 768},
  "linear": {"in_features": 768, "out_features": 2304},
  "attention": {"num_heads": 12, "dropout": 0.0}
}
```

### Optimizer Configuration
```json
{
  "adamw": {
    "lr": 6e-4,
    "betas": [0.9, 0.95],
    "eps": 1e-8
  }
}
```

## Development

### Project Structure
```
.
├── main.py                    # FastAPI application
├── neural_net_model.py        # Model implementation
├── neural_net_layers.py       # Custom layers
├── ddp.py                     # Distributed training
├── loaders.py                 # Dataset management
├── mappers.py                 # Config to PyTorch mapper
├── tokenizers.py              # Tokenization wrapper
├── requirements.txt           # Python dependencies
├── log_config.json           # Logging configuration
├── static/                    # Static web assets
├── templates/                 # HTML templates
├── test_main.py              # API endpoint tests
├── test_neural_net_model.py  # Model tests
├── test_neural_net_layers.py # Layer tests
├── test_loaders.py           # Loader tests
├── test_mappers.py           # Mapper tests
├── test_tokenizers.py        # Tokenizer tests
└── test_ddp.py               # DDP tests
```

### Key Design Decisions

1. **Async Training**: Training runs in background process to avoid blocking API
2. **Locking**: Per-model and per-dataset locks prevent concurrent operations
3. **Gzip Support**: Middleware for compressed request payloads
4. **Shared Memory**: Fast model caching in `/dev/shm` for DDP workers
5. **Error Handling**: Global exception handlers for consistent error responses

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `block_size`
- Use fewer/smaller model layers

### Training Lock Conflict (409 error)
- Wait for current training to complete
- Check `/progress/` endpoint for status
- Delete and recreate model if stuck

### Dataset Download Issues
- Verify HuggingFace dataset path is correct
- Check network connectivity
- Ensure sufficient disk space for shards

## Resources

- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Andrej Karpathy's Neural Networks Course](https://github.com/karpathy/nn-zero-to-hero)
