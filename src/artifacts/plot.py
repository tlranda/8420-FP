import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import json
import os

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--trains', nargs='*', default=None, required=True, type=str, help='Parsed training data in JSON format')
    prs.add_argument('--evals', nargs='*', default=None, required=True, type=str, help='Parsed evaluation data in JSON format')
    prs.add_argument('--save', default=None, type=str, help='Prefix to save images to (default to direct display figures)')
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
            d[train_name(os.path.basename(fname))] = json.load(fin)
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


def time_per_epoch_plot(series, **kwargs):
    fig = plt.figure()
    def time_to_seconds(s):
        fields = s.split(':')
        values = [1, 60, 60, 24, 7, 52]
        scalar = 1
        time = 0
        for f, v in zip(fields[::-1], values):
            scalar *= v
            time += scalar * int(f)
        return time
    choice = lambda x, y: kwargs[x] if x in kwargs.keys() else y
    nth_choice = lambda x, n: kwargs[x][n] if x in kwargs.keys() else n
    if type(series[0]) is list:
        for idx, miniseries in enumerate(series):
            seconds = [time_to_seconds(_) for _ in miniseries]
            epochs = [_ for _ in range(len(seconds))]
            plt.plot(epochs, seconds, label=nth_choice('series_name', idx))
        plt.legend()
    else:
        seconds = [time_to_seconds(_) for _ in series]
        epochs = [_ for _ in range(len(seconds))]
        plt.plot(epochs, seconds)
    plt.xlabel(choice('xlabel', 'Epoch'))
    plt.ylabel(choice('ylabel', 'Epoch Duration (s)'))
    plt.title(choice('title', 'Training Time per Epoch'))
    return fig

def examples_per_epoch_plot(series, **kwargs):
    fig = plt.figure()
    choice = lambda x, y: kwargs[x] if x in kwargs.keys() else y
    nth_choice = lambda x, n: kwargs[x][n] if x in kwargs.keys() else n
    if type(series[0]) is list:
        for idx, miniseries in enumerate(series):
            epochs = [_ for _ in range(len(miniseries))]
            plt.plot(epochs, miniseries, label=nth_choice('series_name', idx))
        plt.legend()
    else:
        epochs = [_ for _ in range(len(series))]
        plt.plot(epochs, series)
    plt.xlabel(choice('xlabel', 'Epoch'))
    plt.ylabel(choice('ylabel', 'Training Examples Seen'))
    plt.title(choice('title', 'Training Examples per Epoch'))
    return fig

def loss_per_epoch_plot(series, **kwargs):
    fig = plt.figure()
    choice = lambda x, y: kwargs[x] if x in kwargs.keys() else y
    nth_choice = lambda x, n: kwargs[x][n] if x in kwargs.keys() else n
    if type(series[0]) is list:
        for idx, miniseries in enumerate(series):
            epochs = [_ for _ in range(len(miniseries))]
            plt.plot(epochs, miniseries, label=nth_choice('series_name', idx))
        plt.legend()
    else:
        epochs = [_ for _ in range(len(series))]
        plt.plot(epochs, series)
    plt.xlabel(choice('xlabel', 'Epoch'))
    plt.ylabel(choice('ylabel', 'Loss'))
    plt.title(choice('title', 'Loss per Epoch'))
    return fig

def score_per_epoch_plot(series, **kwargs):
    fig = plt.figure()
    choice = lambda x, y: kwargs[x] if x in kwargs.keys() else y
    nth_choice = lambda x, n: kwargs[x][n] if x in kwargs.keys() else n
    if type(series[0]) is list:
        for idx, miniseries in enumerate(series):
            epochs = [_ for _ in range(len(miniseries))]
            plt.plot(epochs, miniseries, label=nth_choice('series_name', idx))
        plt.legend()
    else:
        epochs = [_ for _ in range(len(series))]
        plt.plot(epochs, series)
    plt.xlabel(choice('xlabel', 'Epoch'))
    plt.ylabel(choice('ylabel', 'BLEU Score'))
    plt.title(choice('title', 'BLEU Score per Epoch'))
    return fig

def plot(data):
    figs = []
    funcs = [time_per_epoch_plot, examples_per_epoch_plot, loss_per_epoch_plot, score_per_epoch_plot]
    value_keys = ['time', 'seen', 'loss', 'score']
    extra_args = [{}, {}, {}, {}]
    names = list(data.keys())
    for func, key, args in zip(funcs, value_keys, extra_args):
        args['series_name'] = names
        figs.append(func([[vv[key] for vv in v.values()] for k,v in data.items()], **args))
    return figs

def main(args):
    print(args)
    all_data = train_eval_merge(train_merge(args.trains), eval_merge(args.evals))
    figures = plot(all_data)
    if args.save is None:
        plt.show()
    else:
        for fig in figures:
            prefix = args.save+'_'
            outName = prefix+fig.axes[0].title.properties()['text'].replace(' ', '_')+'.png'
            print(outName)
            fig.savefig(outName, format='png')

if __name__ == '__main__':
    main(parse(build()))

