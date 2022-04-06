#!/bin/bash

# Modules
module add anaconda3/2021.05-gcc cuda/11.1.0-gcc
# Virtual Environment
conda create -n 8420 -y
conda activate 8420

# Pytorch Dependency
pip3 install --user torch
# Validation
python -c "import torch; print('Torch Install OK?',torch.cuda.is_available())"

# Huggingfaces Dependency
pip3 install --user datasets
# Validation
python -c "import datasets; print('HuggingFace OK?','wmt14' in datasets.list_datasets())"

# Huggingface Dataset download for WMT14
echo "Downloading HuggingFace Dataset for WMT14 German-English. This may take a moment."
python -c "import datasets; wmt14 = datsets.load_dataset('wmt14', 'de-en')"

# SacreBLEU Dependency for BLEU scores
pip3 install --user sacrebleu
# Validation
python -c "import datasets; print('SacreBleu OK?', datasets.load_metric('sacrebleu') is not None)"

