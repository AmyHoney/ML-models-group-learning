#!/usr/bin/env python3

from transformers import AutoModelForCTC, AutoProcessor
import os

# Fix: requests.exceptions.SSLError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url....aused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1129)'
os.environ['HTTP_PROXY'] = 'http://proxy.vmware.com:3128'
os.environ['HTTPS_PROXY'] = 'http://proxy.vmware.com:3128'


modelname = "facebook/wav2vec2-base-960h"
model = AutoModelForCTC.from_pretrained(modelname)
processor = AutoProcessor.from_pretrained(modelname)

modelpath = "model"
os.makedirs(modelpath, exist_ok=True)
model.save_pretrained(modelpath)
processor.save_pretrained(modelpath)
