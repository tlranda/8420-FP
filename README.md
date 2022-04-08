# 8420-FP
Spring 2022 under Dr. Kai Liu @ Clemson SC

## First-Time Setup
To set up your environment and download all necessary data, just ONCE, execute `. first_setup.sh`.
The script includes a few outputs to ensure that everything runs as expected.

## Future Setups
Prior to running the scripts, execute `. setup.sh` to ensure your environment is ready to run.
The script repeats validation output to ensure everything is OK.

## Short notes:
[SacreBLEU](https://github.com/mjpost/sacreBLEU) has useful examples in the main README to show how to PROPERLY evaluate scores

Google BERT L pretrained model (as a baseline) has test evaluation as follows:
{'score': 19.75294850048873,
 'counts': [34080, 15866, 8704, 5005],
 'totals': [66956, 63953, 60951, 57949],
 'precisions': [50.89909791504869, 24.808843994808687, 14.280323538580172, 8.636904864622341],
 'bp': 0.9943258550353491,
 'sys_len': 66956,
 'ref_len': 67337}
real    36m33.081s

## Roadmap:
1. Setup
   + [:white_check_mark:] Create environment and automate gathering Python dependencies etc as much as possible
   + [:white_check_mark:] Gather datasets (potentially perform some kind of dataset analyses, but hopefully not)
3. Evaluation Pipeline
   + [:white_check_mark:] Using a dummy model that can "cheat," show how the dataset can be loaded, iterated for training
   + [:white_check_mark:] Get semantic evaluation of quality (BLEU score) for the dummy model's predictions
   + [:arrows_counterclockwise:] Get hardware metrics about performance of the dummy model (memory consumption, FLOP/s, walltime, cache utilization, etc)
5. Trainable Transformers
   + [:orange_square:] Basic transformer model (may be template from a repository)
   + [:orange_square:] Reformer transformer model (may be from paper source)
   + [:red_square:] Ablative Reformers (should serve as steps between Basic and Reformer, highlighting individual techniques or gradual compositions)
6. Evaluation and Report
   + [:pushpin:] Use Pipeline to get semantic and hardware metrics for all built transformers
   + [:chart_with_upwards_trend:] Generate meaningful tables, graphics to explain key insights from ablative study
   + [:briefcase:] Deliver non-code items based on this information
