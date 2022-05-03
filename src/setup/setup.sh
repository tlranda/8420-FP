#!/bin/bash

# Modules
module add anaconda3/2021.05-gcc cuda/11.1.0-gcc
# Virtual Environment
conda activate 8420

# Pytorch Dependency
# Validation
python -c "import torch; print('Torch Install ready for GPU?',torch.cuda.is_available())"

# Huggingfaces Dependency
# Validation
python -c "import datasets; print('Datasets OK?','wmt14' in datasets.list_datasets())"
python -c "import transformers; print('HuggingFace Transformers OK?',transformers.is_torch_available())"

# WMT-14 Dataset
echo "Validate presence of WMT14 de-en"
python -c "import datasets; print(datasets.load_dataset('wmt14','de-en'))"

# SacreBLEU Dependency for BLEU scores
# Validation
python -c "import datasets; print('SacreBleu OK?', datasets.load_metric('sacrebleu') is not None)"

# TQDM for training bars
# Validation
python -c "import tqdm; print('TQDM version', tqdm.__version__)"

