import asyncio
import gzip
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app, model_locks

client = TestClient(app, raise_server_exceptions=False)

@pytest.fixture
def mock_new_model():
    with patch("main.NeuralNetworkModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_deserialized_model():
    with patch("neural_net_model.NeuralNetworkModel.deserialize") as mock_deserialize:
        mock_instance = MagicMock()
        mock_deserialize.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_delete_model():
    with patch("neural_net_model.NeuralNetworkModel.delete") as mock_delete:
        yield mock_delete

def test_redirect_to_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert response.url.path == "/dashboard"

def test_create_model_endpoint(mock_new_model):
    payload = {
        "model_id": "test",
        "layers": [
            {"linear": {"in_features": 9, "out_features": 9}, "xavier_uniform": {}, "confidence": 0.9},
            {"sigmoid": {}},
        ] * 2,
        "optimizer": {"sgd": {"lr": 0.1}},
        "device": "cpu",
    }

    response = client.post("/model/", json=payload)

    assert response.status_code == 200, response.json()

    assert response.json() == {
        "message": "Model test created and saved successfully"
    }

    mock_new_model.serialize.assert_called_once()

@pytest.mark.parametrize("input_data, target, output, cost", [
    ([0.0, 0.0, 0.0], None, [0.0, 1.0, 0.0], None),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 1.234),
    ([0.0, 0.0, 0.0], 1, [0.0, 1.0, 0.0], 1.234),
    ([0.0, 0.0, 0.0] * 2, None, [0.0, 1.0, 0.0] * 2, None),
    ([0.0, 0.0, 0.0] * 2, [0.0, 0.0, 1.0] * 2, [0.0, 1.0, 0.0] * 2, 1.234),
    ([0.0, 0.0, 0.0] * 2, [1] * 2, [0.0, 1.0, 0.0] * 2, 1.234),
])
def test_output_endpoint(mock_deserialized_model, input_data, target, output, cost):
    mock_deserialized_model.compute_output.return_value = (output, cost)

    payload = {
        "model_id": "test",
        "input": input_data,
        "target": target,
    }

    response = client.post("/output/", json=payload)

    assert response.json() == {
        "output": output,
        "cost": cost,
    }

    assert response.status_code == 200

@pytest.mark.parametrize("epochs, batch_size, cost", [
    (2, 2, 1.234),
    (3, 3, 1.234),
])
def test_evaluate_endpoint(mock_deserialized_model, epochs, batch_size, cost):
    mock_deserialized_model.evaluate_model.return_value = cost

    payload = {
        "model_id": "test",
        "dataset_id": "mock_ds",
        "shard": 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "block_size": 16,
    }

    response = client.post("/evaluate/", json=payload)

    assert response.json() == {
        "cost": cost,
    }

    assert response.status_code == 200

def test_evaluate_endpoint_with_gzip(mock_deserialized_model):
    cost = 1.234
    mock_deserialized_model.evaluate_model.return_value = cost

    payload = {
        "model_id": "test",
        "dataset_id": "mock_ds",
        "shard": 0,
        "epochs": 3,
        "batch_size": 3,
        "block_size": 16,
    }

    compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))

    response = client.post("/evaluate/", content=compressed_payload,
                           headers={"Content-Encoding": "gzip","Content-Type": "application/json"})

    assert response.json() == {
        "cost": cost,
    }

    assert response.status_code == 200

@pytest.mark.parametrize("input_context, block_size, max_new_tokens, tokens", [
    ([[0]], 8, 2, [0, 1, 2]),
    ([[0, 1]], 4, 2, [0, 1, 2, 3]),
])
def test_generate_endpoint(mock_deserialized_model, input_context, block_size, max_new_tokens, tokens):
    mock_deserialized_model.generate_tokens.return_value = tokens

    payload = {
        "model_id": "test",
        "input": input_context,
        "block_size": block_size,
        "max_new_tokens": max_new_tokens,
    }

    response = client.post("/generate/", json=payload)

    assert response.json() == {
        "tokens": tokens,
    }

    assert response.status_code == 200

def test_train_endpoint(mock_deserialized_model):
    payload = {
        "model_id": "test",
        "dataset_id": "mock_ds",
        "shard": 1,
        "epochs": 2,
        "batch_size": 1,
        "block_size": 3,
    }

    response = client.put("/train/", json=payload)

    assert response.status_code == 202
    assert response.json() == {"message": "Training for model test started asynchronously."}

def test_train_endpoint_returns_409_when_already_locked(mock_deserialized_model):
    payload = {
        "model_id": "test",
        "dataset_id": "mock_ds",
        "shard": 1,
        "epochs": 2,
        "batch_size": 1,
        "block_size": 3,
    }

    lock = asyncio.Lock()
    model_locks["test"] = lock
    # Manually acquire the lock
    asyncio.run(lock.acquire())

    # Now when we send request, it should see lock.locked() == True
    response = client.put("/train/", json=payload)

    assert response.status_code == 409
    assert response.json() == {"detail": "Training already in progress for model test."}

    # Clean up after test
    del model_locks["test"]

def test_train_endpoint_with_gzip(mock_deserialized_model):
    payload = {
        "model_id": "test",
        "dataset_id": "mock_ds",
        "shard": 1,
        "epochs": 2,
        "batch_size": 1,
        "block_size": 3,
    }

    compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))

    response = client.put("/train/", content=compressed_payload,
                          headers={"Content-Encoding": "gzip","Content-Type": "application/json"})

    assert response.status_code == 202
    assert response.json() == {"message": "Training for model test started asynchronously."}

def test_progress_endpoint(mock_deserialized_model):
    mock_deserialized_model.progress = [
        "Some progress"
    ]
    mock_deserialized_model.avg_cost = 0.123
    mock_deserialized_model.avg_cost_history = [0.1, 0.2, 0.3]
    mock_deserialized_model.status = "Teapot ;-)"

    response = client.get("/progress/", params={"model_id": "test"})

    assert response.status_code == 200

    assert response.json() == {
        "progress": [
            "Some progress"
        ],
        "average_cost": 0.123,
        "average_cost_history": [0.1, 0.2, 0.3],
        "status": "Teapot ;-)"
    }

def test_stats_endpoint(mock_deserialized_model):
    mock_deserialized_model.stats = {
        "some": "stats",
    }

    response = client.get("/stats/", params={"model_id": "test"})

    assert response.status_code == 200

    assert response.json() == {
        "some": "stats",
    }

def test_not_found(mock_deserialized_model):
    mock_deserialized_model.compute_output.side_effect = KeyError("Testing key error :-)")

    response = client.post("/output/", json={
        "model_id": "nonexistent",
        "input": [0, 0, 0],
    })

    assert response.status_code == 404
    assert response.json() == {'detail': "Not found error occurred: 'Testing key error :-)'"}

def test_invalid_payload():
    response = client.post("/output/", json={
        "model_id": "test",
        # Missing "input" key
    })

    assert response.status_code == 422
    assert "detail" in response.json()

def test_value_error(mock_deserialized_model):
    mock_deserialized_model.compute_output.side_effect = ValueError("Testing value error :-)")

    response = client.post("/output/", json={
        "model_id": "test",
        "input": [0, 0, 0],
    })

    assert response.status_code == 400
    assert response.json() == {'detail': 'Value error occurred: Testing value error :-)'}

def test_unhandled_exception(mock_deserialized_model):
    mock_deserialized_model.compute_output.side_effect = RuntimeError("Unexpected error")

    response = client.post("/output/", json={
        "model_id": "test",
        "input": [0, 0, 0],
    })

    assert response.status_code == 500
    assert response.json() == {"detail": "Please refer to server logs"}

def test_delete_model_endpoint(mock_delete_model):
    response = client.delete("/model/", params={"model_id": "test"})

    assert response.status_code == 204

    mock_delete_model.assert_called_once()

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
