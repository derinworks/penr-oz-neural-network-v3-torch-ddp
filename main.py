from __future__ import annotations
import logging
import math
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.params import Query
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import multiprocessing as mp
from pydantic import BaseModel, Field
from asyncio import Lock, create_task
from typing import Dict
import gzip
import ddp
from mappers import Mapper
from neural_net_model import NeuralNetworkModel
from tokenizers import Tokenizer
from loaders import Downloader, Loader

app = FastAPI(
    title="Neural Network Model API v3",
    description="API to create, serialize, output, evaluate, generate, train and diagnose of neural network models.",
    version="0.3.0"
)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

log = logging.getLogger(__name__)

# This will track active sessions locks by id
dataset_locks: Dict[str, Lock] = {}
model_locks: Dict[str, Lock] = {}


class ModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["gpt-example"],
        description="The unique identifier for the model."
    )

class ModelOnDeviceRequest(ModelRequest):
    device: str = Field(
        "cpu",
        examples=["cpu", "cuda"],
        description="A device name for PyTorch to move the model to (default: cpu)"
    )


class CreateModelRequest(ModelRequest):
    layers: list[dict] = Field(
        ...,
        examples=[
            [{"summation": [{"embedding": {"num_embeddings": 50304, "embedding_dim": 768},
                             "normal": {"mean": 0.0, "std": 0.02}},
                            {"position": {"num_embeddings": 1024, "embedding_dim": 768},
                             "normal": {"mean": 0.0, "std": 0.02}}]},
             {"dropout": {"p": 0.0}}] +
            [{"residual": [{"sequential": [{"layernorm": {"normalized_shape": 768}},
                                           {"linear": {"in_features": 768, "out_features": 3 * 768},
                                            "normal": {"mean": 0.0, "std": 0.02},
                                            "zeros": {}},
                                           {"attention": {"num_heads": 12, "dropout": 0.0}},
                                           {"linear": {"in_features": 768, "out_features": 768},
                                            "normal": {"mean": 0.0, "std": 0.02 / math.sqrt(2 * 12)},
                                            "zeros": {}},
                                           {"dropout": {"p": 0.0}}]},
                           {"sequential": [{"layernorm": {"normalized_shape": 768}},
                                           {"linear": {"in_features": 768, "out_features": 4 * 768},
                                            "normal": {"mean": 0.0, "std": 0.02},
                                            "zeros": {}},
                                           {"gelu": {}},
                                           {"linear": {"in_features": 4 * 768, "out_features": 768},
                                            "normal": {"mean": 0.0, "std": 0.02 / math.sqrt(2 * 12)},
                                            "zeros": {}},
                                           {"dropout": {"p": 0.0}}]}
                           ]} for _ in range(12)] +
            [{"layernorm": {"normalized_shape": 768}},
             {"linear": {"in_features": 768, "out_features": 50304, "bias": False}},
             {"softmaxlast": {"dim": -1}}]
        ],
        description="List of dictionaries where the key is the PyTorch nn or init algorithm and the value is its args."
    )
    optimizer: dict = Field(
        ...,
        examples=[
            {"adamw": {"lr": 6e-4, "betas": [0.9, 0.95], "eps": 1e-8}}
        ],
        description="A dictionary where the key is the PyTorch optimizer name and the value is its args."
    )

class DatasetRequest(BaseModel):
    dataset_id: str = Field(
        ...,
        examples=["tiny-shakespeare"],
        description="The unique identifier for the dataset"
    )

class TokenizerRequest(BaseModel):
    encoding: str = Field(
        ...,
        examples=["gpt2"],
        description="Tiktoken encoding to use"
    )

class DownloadDatasetRequest(DatasetRequest, TokenizerRequest):
    path: str = Field(
        ...,
        examples=["andriotis/tiny-shakespeare-karpathy"],
        description="A remote path to the HuggingFace dataset to download"
    )
    name: str = Field(
        ...,
        examples=["default"],
        description="A remote name of the HuggingFace dataset to download"
    )
    split: str = Field(
        ...,
        examples=["train"],
        description="Split of the HuggingFace dataset to download"
    )
    shard_size: int = Field(
        ...,
        examples=[1e5],
        description="Number of tokens per shard"
    )

class TrainingRequest(ModelOnDeviceRequest, DatasetRequest):
    shard: int = Field(
        ...,
        examples=[1],
        description="Specified a shard of the dataset to begin training from"
    )
    epochs: int = Field(
        ...,
        examples=[4],
        description="The number of training epochs"
    )
    batch_size: int = Field(
        ...,
        examples=[2],
        description="The batch size to sample each epoch"
    )
    block_size: int = Field(
        ...,
        examples=[1024],
        description="The block size (or sequence length) of each single sample entry in a batch"
    )

class EvaluateRequest(TrainingRequest):
    target_dataset_id: str | None = Field(
        None,
        examples=[None],
        description="Separate target dataset to use for evaluation (Optional)"
    )
    shard: int = Field(
        ...,
        examples=[0],
        description="Specified a shard of the dataset(s) to begin evaluation from"
    )
    epochs: int = Field(
        ...,
        examples=[2],
        description="The number of evaluation epochs"
    )

class TokenizeTextRequest(TokenizerRequest):
    text: str = Field(
        ...,
        examples=[
            """
            PENR-OZ:
            I say Hello world!
            """
        ],
        description="Text to tokenize to prepare as input for neural network"
    )

class OutputRequest(ModelRequest):
    input: list = Field(
        ...,
        examples=[
            [[0]]
        ],
        description="The initial input context"
    )
    target: list | int | None = Field(
        None,
        examples=[None],
        description="The expected target data (Optional)"
    )

class GenerateRequest(ModelRequest):
    input: list = Field(
        ...,
        examples=[
            [[0]]
        ],
        description="The initial input context"
    )
    block_size: int = Field(
        ...,
        examples=[1024],
        description="The block size of context"
    )
    max_new_tokens: int = Field(
        ...,
        examples=[10],
        description="The maximum number of tokens to generate"
    )
    temperature: float = Field(
        1.0,
        examples=[1.0],
        description="The temperature ratio for logits"
    )
    top_k: int | None = Field(
        None,
        examples=[None],
        description="Use Top K results"
    )

class DecodeTokensRequest(TokenizerRequest):
    tokens: list[int] = Field(
        ...,
        examples=[[0]],
        description="Previously Tiktoken encoded or generated tokens"
    )

class ModelIdQuery(Query):
    description="The unique identifier for the model"

class DatasetIdQuery(Query):
    description="The unique identifier for the dataset"

@app.middleware("http")
async def gzip_decompression_middleware(request: Request, call_next):
    if request.headers.get("Content-Encoding", "").lower() == "gzip":
        body = await request.body()
        log.info(f"Retrieved gzip encoded request body")
        decompressed_body = gzip.decompress(body)
        log.info(f"Decompressed gzip encoded body")
        request._body = decompressed_body
        async def decompressed_receive(): # pragma: no cover
            return {"type": "http.request", "body": decompressed_body, "more_body": False}
        request._receive = decompressed_receive
    return await call_next(request)

@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, e: Exception):
    log.error(f"An error occurred: {str(e)}")
    return JSONResponse(status_code=500, content={"detail": "Please refer to server logs"})

@app.exception_handler(KeyError)
async def key_error_handler(_: Request, e: KeyError):
    raise HTTPException(status_code=404, detail=f"Not found error occurred: {str(e)}")

@app.exception_handler(ValueError)
async def value_error_handler(_: Request, e: ValueError):
    raise HTTPException(status_code=400, detail=f"Value error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.post("/model/")
def create_model(body: CreateModelRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting creation of model {model_id}")
    model = NeuralNetworkModel(model_id, Mapper(body.layers, body.optimizer))
    model.serialize()
    return {"message": f"Model {model_id} created and saved successfully"}

@app.get("/dataset/")
def list_dataset(dataset_id: str = DatasetIdQuery(...)):
    log.info(f"Requesting list of files for dataset {dataset_id}")
    loader = Loader(dataset_id)
    files = loader.list()
    return {
        "files": files
    }

@app.post("/dataset/")
async def download_dataset(body: DownloadDatasetRequest = Body(...)):
    dataset_id = body.dataset_id
    log.info(f"Requesting download of dataset {dataset_id}")

    # Get or create a lock for this dataset
    if dataset_id not in dataset_locks:
        dataset_locks[dataset_id] = Lock()
    lock = dataset_locks[dataset_id]

    # If the model is already locked (downloading), return 409 Conflict
    if lock.locked():
        raise HTTPException(status_code=409, detail=f"Downloading dataset {dataset_id} already in progress.")

    downloader = Downloader(dataset_id, body.shard_size, body.encoding)

    async def download():
        async with lock:
            await run_in_threadpool(downloader.download,body.path, body.name, body.split)

    # Start training in the background
    create_task(download())

    # Respond with request accepted
    return JSONResponse(content={"message": f"Downloading Dataset {dataset_id} asynchronously."}, status_code=202)

@app.delete("/dataset/")
def delete_dataset(dataset_id: str = DatasetIdQuery(...)):
    log.info(f"Requesting deletion of dataset {dataset_id}")
    loader = Loader(dataset_id)
    loader.delete()
    return Response(status_code=204)

@app.post("/tokenize/")
def tokenize_text(body: TokenizeTextRequest = Body(...)):
    log.info(f"Requesting tokenization of text {body.text}")
    tokenizer = Tokenizer(body.encoding)
    tokens = tokenizer.tokenize(body.text)
    return {"encoding": body.encoding,
            "tokens": tokens}

@app.post("/output/")
def compute_model_output(body: OutputRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting output for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    output, cost = model.compute_output(body.input, body.target)
    return {"output": output,
            "cost": cost,
            }

@app.post("/evaluate/")
def evaluate_model(body: EvaluateRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Requesting evaluation of model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    cost = model.evaluate_model(body.dataset_id, body.target_dataset_id, body.shard,
                                body.epochs, body.batch_size, body.block_size)
    return {"cost": cost}

@app.post("/generate/")
def model_generate(body: GenerateRequest = Body(...)):
    model_id = body.model_id
    log.info(f"Generating tokens using model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    generated_tokens = model.generate_tokens(body.input, body.block_size, body.max_new_tokens,
                                             body.temperature, body.top_k)
    return {"tokens": generated_tokens}

@app.post("/decode/")
def decode_tokens(body: DecodeTokensRequest = Body(...)):
    log.info(f"Requesting decoding of {len(body.tokens)} token(s)")
    tokenizer = Tokenizer(body.encoding)
    decoded_text = tokenizer.decode(body.tokens)
    return {"encoding": body.encoding,
            "text": decoded_text}

@app.put("/train/")
async def train_model(body: TrainingRequest = Body(...)):
    model_id = body.model_id
    device = body.device
    log.info(f"Requesting training for model {model_id} on device {device}")

    # Get or create a lock for this model
    if model_id not in model_locks:
        model_locks[model_id] = Lock()
    lock = model_locks[model_id]

    # If the model is already locked (training), return 409 Conflict
    if lock.locked():
        raise HTTPException(status_code=409, detail=f"Training already in progress for model {model_id}.")

    async def _launch():
        async with lock:
            ddp_launch_proc = mp.Process(target=ddp.launch_single_node_ddp, args=(
                model_id, device, NeuralNetworkModel.train_model_on_device, model_id, device,
                body.dataset_id, body.shard, body.epochs, body.batch_size, body.block_size))
            ddp_launch_proc.start()
            log.info(f"Waiting for distributed training process for model {model_id} to complete...")
            await run_in_threadpool(ddp_launch_proc.join)
            log.info(f"Distributed training process completed for model {model_id}")

    # Start training in the background
    create_task(_launch())

    # Respond with request accepted
    return JSONResponse(content={"message": f"Training for model {model_id} started asynchronously."}, status_code=202)

@app.get("/progress/")
def model_progress(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting progress for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    return {
        "progress": model.progress,
        "average_cost": model.avg_cost,
        "average_cost_history": model.avg_cost_history,
        "status": model.status,
    }

@app.get("/stats/")
def model_stats(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting stats for model {model_id}")
    model = NeuralNetworkModel.deserialize(model_id)
    return model.stats

@app.delete("/model/")
def delete_model(model_id: str = ModelIdQuery(...)):
    log.info(f"Requesting deletion of model {model_id}")
    NeuralNetworkModel.delete(model_id)
    return Response(status_code=204)


if __name__ == "__main__": # pragma: no cover
    import uvicorn
    import json

    with open("log_config.json", "r") as f:
        log_config = json.load(f)

    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=log_config)
