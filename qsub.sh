#!/bin/bash

#PBS -N 8420_TRAIN
#PBS -l select=1:ncpus=4:mem=100gb:ngpus=1:gpu_model=v100,walltime=50:00:00

hostname; date;

# CD here (omitted)

module add anaconda3/2021.05-gcc cuda/11.1.0-gcc;
conda activate 8420;
which python;

date;
# NOTE: ONLY RUN ONE OF THE BELOW PER SCRIPT INVOCATION.
# Training is time-intensive and will take too long to run both in their entirety in the same job
# Recommend submitting two jobs, with one or the other below commented out on each

# Evaluations done without bert embeddings
# Expected runtime: 2 days
python train_pipeline.py --epochs 42 --save saved/attempt_two_ --skip-load bad_batches.txt --skip-save a2_bad.txt --eval-limit 0;
# Evaluations done with bert embeddings
# Expected runtime: 2 days
python train_pipeline.py --epochs 8 --save saved/preBERT_ --skip-save pre_bad.txt --eval-limit 0;

date;

