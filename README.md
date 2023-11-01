# Multimodal Model Cloud Performance Test

## Overview

Performance tests of multimodal models on cloud providers.

## LLaVA on Runpod

[Runpod](https://runpod.io) has generally good availability but is more expensive than Amazon.

[LLaVA](https://github.com/haotian-liu/LLaVA) (the PyTorch repo) was tested using the `llava-test.py` script to connect to a worker process.

**Note:** Very important that the model files be in directories that are named with `llava-v1.5-` in order for the conversation mode to be picked up correctly.

Commands to run on the server (in this order) to bring up the required controller process (redundant but the worker expects to find it) and worker after ensuring
all Python dependencies are installed:

```
    python -m llava.serve.controller --host 0.0.0.0 --port 10000
    python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 8080 --worker http://localhost:8080 --model-path models/llava-v1.5-7b

```

Then, copy all the images in this repo's `images` directory along with `llava-test.py` to the LLaVA repo root directory and run:

```
    python llava-test.py
```