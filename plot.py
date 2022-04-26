import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import json
import pdb

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--trains', nargs='*', default=None, required=True, type=str, help='Parsed training data in JSON format')
    prs.add_argument('--evals', nargs='*', default=None, required=True, type=str, help='Parsed evaluation data in JSON format')
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

# Collide names in the parsing dictionaries
def train_name(fname):
    return fname[:fname.index('train_parsed.json')]
def eval_name(fname):
    return fname[:fname.index('eval_parsed.json')]
# Merge a number of parsed training files
def train_merge(tli):
    d = {}
    for fname in tli:
        with open(fname, 'r') as fin:
            d[train_name(fname)] = json.load(fin)
    return d
# Merge a number of parsed evaluation files
def eval_merge(eli):
    d = {}
    for fname in eli:
        with open(fname, 'r') as fin:
            fd = json.load(fin)
            for k,v in fd.items():
                if k in d.keys():
                    raise Exception(f"Multiple evaluation keys for {k}")
                d[k] = v
    return d
# Forcibly collapse training and evaluation data into the same structure
def train_eval_merge(ts, es):
    merged = {}
    for tr_key in ts.keys():
        if tr_key in es.keys():
            merged[tr_key] = ts[tr_key]
            for epoch_key in ts[tr_key].keys():
                merged[tr_key][epoch_key]['score'] = es[tr_key][int(epoch_key)-1]
    return merged

def plot(data):
    pass

def main(args):
    print(args)
    all_data = train_eval_merge(train_merge(args.trains), eval_merge(args.evals))
    plot(all_data)

if __name__ == '__main__':
    main(parse(build()))

