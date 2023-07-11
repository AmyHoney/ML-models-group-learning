#!/bin/bash
set -eu pipefail

mkdir -p model_store
# Extra files add all files necessary for processor
torch-model-archiver --model-name red --version 1.0 --handler custom_handler.py --extra-files model.zip,setup_config.json -f
mv red.mar model_store

echo "Archive model MAR package finished!"