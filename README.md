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

## Roadmap:
1. Setup
   + [:white_check_mark:] Create environment and automate gathering Python dependencies etc as much as possible
   + [:white_check_mark:] Gather datasets (potentially perform some kind of dataset analyses, but hopefully not)
3. Evaluation Pipeline
   + [:arrows_counterclockwise:] Using a dummy model that can "cheat," show how the dataset can be loaded, iterated for training
   + [:soon:] Get semantic evaluation of quality (BLEU score) for the dummy model's predictions
   + [:soon:] Get hardware metrics about performance of the dummy model (memory consumption, FLOP/s, walltime, cache utilization, etc)
5. Transformers
   + [:orange_square:] Basic transformer model (may be template from a repository)
   + [:orange_square:] Reformer transformer model (may be from paper source)
   + [:red_square:] Ablative Reformers (should serve as steps between Basic and Reformer, highlighting individual techniques or gradual compositions)
6. Evaluation and Report
   + [:pushpin:] Use Pipeline to get semantic and hardware metrics for all built transformers
   + [:chart_with_upwards_trend:] Generate meaningful tables, graphics to explain key insights from ablative study
   + [:briefcase:] Deliver non-code items based on this information
