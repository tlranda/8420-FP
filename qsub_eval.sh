#!/bin/bash

#PBS -N 8420_TRAIN
#PBS -l select=1:ncpus=4:mem=100gb:ngpus=1:gpu_model=v100,walltime=48:00:00

hostname; date;

# CD here (omitted)

module add anaconda3/2021.05-gcc cuda/11.1.0-gcc;
conda activate 8420;
which python;

date;
python batch_eval.py --dir saved;
date;


