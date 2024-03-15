#! /bin/bash

if [ ! -d "pretrained_models_ckpts" ]; then
    echo "Downloading pretrained models..."
    make download_pretrained_models
fi

if [ -f .env ]; then
    export $(cat .env | xargs)
fi

source venv/bin/activate
TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node $2 \
                                      src/iamcl2r/main.py -c $1
