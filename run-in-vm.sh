#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --log-config log_config.json --host 0.0.0.0
