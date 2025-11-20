# penr-oz-neural-network-v3-torch-ddp
Version 3 Implementation of Neural Network service leveraging pytorch distributed data platform (ddp)

This repository demonstrates same key concepts in neural networks as in [penr-oz-neural-network-v2-torch-nn](https://github.com/derinworks/penr-oz-neural-network-v2-torch-nn) 
with automatic gradient descent calculations relying on PyTorch library leveraging Neural Network (nn) package, 
Distributed Data Parallel (ddp) feature to support scaling to multiple GPU (CUDA) devices, and changes to API to support
downloading/sharding training data for local read instead of an API payload.

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
