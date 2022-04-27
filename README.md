# 8420-FP
Spring 2022 under Dr. Kai Liu @ Clemson SC

## Instructions for Replication:
### 1. Set up your environment.
Run the `src/setup/first_setup.sh` script (configured for Palmetto cluster with their modulesystem) to create your environment and install all necessary software.
The script includes minor validation along the way to aid in troubleshooting.
Future uses can run `src/setup/setup.sh` to skip the installation steps, though troubleshooting validation still runs (can test if your existing environment is compatible with the project by editing the python environment selection in the first few lines).
### 2. Run the training script.
It is recommended to run `qsub.sh` via the PBS scheduler (after editing its omitted CD command to ensure all paths are correct!), but you may reference `python src/transformers/train_pipeline.py --help` for complete argument specification.
If running interactively, all stderr output should be redirected to file for later use.
### 3. During and/or after training, run the evaluation script.
To allow for node-parallel, dynamically scheduled evaluation, run `qsub_eval.sh` (again, after editing its omitted CD command to ensure all paths are correct!) to evaluate your training epoch checkpoints.
If running interactively, all stdout output should be redirected to file for later use.
### 4. Parse train and evaluation output.
The saved outputs from steps 2 and 3 should be suitable inputs to `src/artifacts/parse.py` (see `python src/artifacts/parse.py --help` for assistance), which will produce json outputs similar to the ones included in this repository.
Due to the size of the model checkpoints and training error files, the necessary binary and logs to reproduce these JSON outputs are not directly included in the repository, however, sufficiently similar outputs should be replicable by feeding outputs from steps 2 and 3 as described here.
We provide our own JSON files, as they are relatively small, in `src/artifacts/parsed`.
### 5. Produce plots.
The JSON files produced in step 4 should be passed to `src/artifacts/plot.py` (see `python src/artifacts/plot.py --help` for assistance), which willl create the figures used in our deliverables.

## Summary Evaluations:

* The pretrained Google BERT L model (used as a baseline) receives a SacreBLEU score of 16.717185446966926.
* The basic transformer model (without pretrained embeddings) receives a peak SacreBLEU score of 0.001334493082716761.
* The basic transformer model with BERT pretrained embeddings recevies a peak SacreBLEU score of 0.001279603793868831.

## Roadmap:
1. Setup
   + [:white_check_mark:] Create environment and automate gathering Python dependencies etc as much as possible
   + [:white_check_mark:] Gather datasets (potentially perform some kind of dataset analyses, but hopefully not)
3. Evaluation Pipeline
   + [:white_check_mark:] Using a dummy model that can "cheat," show how the dataset can be loaded, iterated for training
   + [:white_check_mark:] Get semantic evaluation of quality (BLEU score) for the dummy model's predictions
   + [:x:] [SCOPE CUT] Get hardware metrics about performance of the dummy model (memory consumption, FLOP/s, walltime, cache utilization, etc)
5. Trainable Transformers
   + [:white_check_mark:] Basic transformer model (use pytorch transformer models)
   + [:white_check_mark:] Basic transformer model with proper embeddings (use pytorch transformer models and pretrained embeddings)
   + [:x:] [SCOPE CUT] Locality Sensitive Hashed Attention
   + [:x:] [SCOPE CUT] Reversible Transformer
   + [:x:] [SCOPE CUT] Chunked Parameter Groups
   + [:x:] [SCOPE CUT] Reformer transformer model (may be from paper source or simply combine all techniques)
6. Evaluation and Report
   + [:pushpin:] Use Pipeline to get semantic and hardware metrics for all built transformers
   + [:chart_with_upwards_trend:] Generate meaningful tables, graphics to explain key insights from ablative study
   + [:briefcase:] Deliver non-code items based on this information

