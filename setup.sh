#!/bin/bash

# Modules
module add anaconda3/2021.05-gcc cuda/11.1.0-gcc
# Virtual Environment
conda activate 8420

# Pytorch Dependency
# Validation
python -c "import torch; print('Torch Install OK?',torch.cuda.is_available())"

# Huggingfaces Dependency
# Validation
python -c "import datasets; print('Datasets OK?','wmt14' in datasets.list_datasets())"

# WMT-14 Dataset
echo "Validate presence of WMT14 de-en"
python -c "import datasets; print(datasets.load_dataset('wmt14','de-en'))"

# SacreBLEU Dependency for BLEU scores
pip3 install --user sacrebleu
# Validation
python -c "import datasets; print('SacreBleu OK?', datasets.load_metric('sacrebleu') is not None)"

