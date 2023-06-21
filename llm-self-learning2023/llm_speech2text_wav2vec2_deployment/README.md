# Speech2text (facebook/wav2vec2-base-960h) model inference with Huggingface API2

## Overview

[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) the base model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz.

## Model inference

### Create and run a Notebook server

Use a customized image that has Java and torchserve installed. You can use [Dockerfile](../Dockerfile) to generate your own custom image. You can also directly use an image published on VMware harbor repo:

```bash
projects.registry.vmware.com/models/notebook/hf-inference-deploy:v1
```

Start the Notebook server with GPU:

```bash
docker run --gpus all -it -v `pwd`:/mnt projects.registry.vmware.com/models/notebook/hf-inference-deploy:v1
```

### Model Inference

```bash
python3 redpajama-inference.py
```
