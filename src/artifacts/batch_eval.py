import os
import argparse
import sys
# Fetch restructured directories to import other code
sys.path.append(os.path.relpath("../transformers"))
from evaluation_pipeline import (load_wmt14,
                                 load_sacrebleu,
                                 evaluate,
                                 torch)
from models import (choices, lookup)
"""
    Evaluate the SacreBLEU score of many models in back-to-back fashion
    Automatically performs some light persistence via cache to improve performance
"""

def main(args):
    print(args)
    if args.device == 'cpu':
        print(f"Warning! Running on CPU!")
    # Fetch dataset
    data = load_wmt14()
    # Load cache of evaluations
    cache_evals = torch.load(args.cache)
    # Iterate through model states to evaluate
    # args.nhid is overwritten by the model.__init__() call, which is pretty bad design but
    # we can fix it here by tracking if it should be reset or not
    reset_nhid = args.nhid is None
    for f in args.files:
        if f.startswith(args.cache_trim):
            cache_f = f[len(args.cache_trim):]
        else:
            cache_f = f
        if cache_f not in cache_evals.keys() and not args.skip_uncached:
            print(f"Load {cache_f} for evaluation")
            if reset_nhid:
                args.nhid = None
            transformer = lookup[args.model](args)
            # Load and apply the model parameters from file
            try:
                params = torch.load(f)
                transformer.load_state_dict(params)
            except RuntimeError:
                print(f"!! Unable to load '{cache_f}' under current parameters. Make sure your provided model arguments match the training scenario")
            else:
                # Create lambda function to allow for evaluation
                translator = lambda de: transformer.evaluate(de)
                # Add to cache
                cache_evals[cache_f] = evaluate(data['test'], translator)
                # Update cache on disk so we don't lose this eval
                torch.save(cache_evals, 'batch_eval_cache.pt')
        if cache_f in cache_evals.keys():
            print(cache_f, cache_evals[cache_f])
        else:
            print(cache_f, " -- NO EVAL YET -- ")

def build():
    prs = argparse.ArgumentParser()
    # Cache behavior
    prs.add_argument('--cache', type=str, default='batch_eval_cache.pt', help='Cache file to utilize (default batch_eval_cache.pt)')
    prs.add_argument('--skip-uncached', action='store_true', help='Only show results in the cache, don\'t compute new ones')
    prs.add_argument('--cache-trim', type=str, default='', help='Prefix to remove from cache entries (default no trimming)')
    # Specify saved models to load
    prs.add_argument('--dir', type=str, nargs='*', default=None, help='Directory to read files from (default None)')
    prs.add_argument('--add', type=str, nargs='*', default=None, help='Additional files to directly include (default None)')
    prs.add_argument('--remove', '--rm', type=str, nargs='*', default=None, help='Files to ignore from compiled lists (default None)')
    # Specify how the model should be loaded
    prs.add_argument('--model', type=str, choices=choices, default=choices[0], help='Model used to train (MUST MATCH FOR ALL FILES)')
    prs.add_argument('--nhid', type=int, default=None, help='Hidden units per layer (default PreTrainedBERT)')
    prs.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    prs.add_argument('--nhead', type=int, default=2, help='Number of heads in encoder/decoder blocks')
    prs.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    prs.add_argument('--dropout', type=float, default=0.2, help='Dropout per layer (0 = no dropout)')
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Flatten add and remove lists
    remove = []
    if args.remove is not None:
        for entry in args.remove:
            remove.append(entry)
    args.remove = set(remove)
    add = []
    if args.add is not None:
        for entry in args.add:
            entry = set([entry]).difference(args.remove)
            add.extend([_ for _ in entry])
    args.files = set(add)
    # Get additional files to consider
    if args.dir is not None:
        for d in args.dir:
            for f in os.listdir(d):
                # Slap directory back on the path
                f = d+'/'+f
                if os.path.isfile(f) and f not in args.remove:
                    args.files.update([f])
    args.files = sorted(args.files)
    del args.add, args.dir, args.remove
    # Select a device
    args.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    # Arguments that need to exist for the thing to not crash, but that we have ZERO INTEREST in controlling
    args.skip_load = None
    return args

if __name__ == '__main__':
    main(parse(build()))

