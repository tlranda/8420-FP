import os
import subprocess
from evaluation_pipeline import (load_wmt14,
                                 load_sacrebleu,
                                 evaluate,
                                 torch)
from models import (choices, lookup)
import argparse
import pdb


def main(args):
    print(args)
    if args.device == 'cpu':
        print(f"Warning! Running on CPU!")
    # Fetch dataset
    data = load_wmt14()
    # Load cache of evaluations
    cache_evals = torch.load('batch_eval_cache.pt')
    # Iterate through model states to evaluate
    for f in args.files:
        if f not in cache_evals.keys() and not args.skip_uncached:
            transformer = lookup[args.model](args)
            params = torch.load(f)
            transformer.load_state_dict(params)
            translator = lambda de: transformer.evaluate(de)
            # Add to cache
            cache_evals[f] = evaluate(data['test'], translator)
            # Update cache on disk so we don't lose this eval
            torch.save(cache_evals, 'batch_eval_cache.pt')
        if f in cache_evals.keys():
            print(f, cache_evals[f])
        else:
            print(f, " -- NO EVAL YET -- ")

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--skip-uncached', action='store_true', help='Only show results in the cache, don\'t compute new ones')
    # Specify saved models to load
    prs.add_argument('--dir', type=str, nargs='*', default=None, help='Directory to read files from (default None)')
    prs.add_argument('--add', type=str, nargs='*', default=None, help='Additional files to directly include (default None)')
    prs.add_argument('--remove', '--rm', type=str, nargs='*', default=None, help='Files to ignore from compiled lists (default None)')
    # Specify how the model should be loaded
    prs.add_argument('--model', type=str, choices=choices, default=choices[0], help='Model used to train (MUST MATCH FOR ALL FILES)')
    prs.add_argument('--nhid', type=int, default=256, help='Hidden units per layer')
    prs.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    prs.add_argument('--nhead', type=int, default=2, help='Number of heads in encoder/decoder blocks')
    prs.add_argument('--lr', type=float, default=20, help='Initial learning rate')
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
