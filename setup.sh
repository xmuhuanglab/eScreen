#!/bin/bash

# create environment
conda env create -n eScreen -f escreen.yaml -y
conda run -n eScreen pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
conda install -n eScreen -c nvidia cuda-nvcc cuda-cudart-dev -y

# Using conda run to conduct all pip install pipeline
conda run -n eScreen bash -c '
    mkdir flash-attn
    cd flash-attn
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    cd ..
    pip install ./flash-attn/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation --no-deps
    pip install ./flash-fft-conv/flashfftconv-0.0.0-py3-none-any.whl
    pip install ./flash-fft-conv/monarch_cuda-0.0.0-cp310-cp310-linux_x86_64.whl
    python -m ipykernel install --user --name eScreen
'
