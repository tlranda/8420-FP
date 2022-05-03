#!/bin/bash

# Modules
module add anaconda3/2021.05-gcc cuda/11.1.0-gcc
# Virtual Environment
conda create -n 8420 -y
conda activate 8420

# Pytorch Dependency
pip3 install --user torch
# Validation
python -c "import torch; print('Torch Install ready for GPU?',torch.cuda.is_available())"

# Huggingfaces Dependency
pip3 install --user datasets
pip3 install --user transformers
# Validation
python -c "import datasets; print('HuggingFace Data OK?','wmt14' in datasets.list_datasets())"
python -c "import transformers; print('HuggingFace Transformers OK?',transformers.is_torch_available())"

# Huggingface Dataset download for WMT14
echo "Downloading HuggingFace Dataset for WMT14 German-English. This may take a moment."
python -c "import datasets; wmt14 = datsets.load_dataset('wmt14', 'de-en')"

# SacreBLEU Dependency for BLEU scores
pip3 install --user sacrebleu
# Validation
python -c "import datasets; print('SacreBleu OK?', datasets.load_metric('sacrebleu') is not None)"

# TQDM for training bars
pip3 install --user tqdm
# Validation
python -c "import tqdm; print('TQDM version', tqdm.__version__)"

