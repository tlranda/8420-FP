import argparse
import os
import json

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--files', nargs='*', default=None, type=str, required=True, help='File(s) to read')
    prs.add_argument('--modes', choices=['auto','train','eval'], default=None, nargs='*', help='What files are parsed (default: auto determine from file name)')
    prs.add_argument('--suffix', type=str, default='_parsed', help='Suffix (before extension, if exists) to apply to parsed output per file (default \'_parsed\')')
    prs.add_argument('--recompute', action='store_true', help='Do not accept pre-parsed files as cache, explicitly compute all inputs')
    prs.add_argument('--not-json', action='store_true', help='Preserve original file extension rather than substituting it with .json')
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Assign modes to each file
    if args.modes is None:
        args.modes = []
    if len(args.modes) < len(args.files):
        args.modes.extend(['auto'] * (len(args.files) - len(args.modes)))
    new_modes = []
    for fname, mode in zip(args.files, args.modes):
        if mode == 'auto':
            if 'eval' in fname.lower():
                new_modes.append('eval')
            elif 'train' in fname.lower():
                new_modes.append('train')
            else:
                print(f"Unable to identify mode for {fname}. Pick [t]rain or [e]val?")
                choice = input()
                while choice not in ['t', 'train', 'e', 'eval']:
                    print(f"Did not recognize '{choice}'. Please pick [t]rain or [e]val for {fname}?")
                    choice = input()
                if choice.startswith('t'):
                    new_modes.append('train')
                else:
                    new_modes.append('eval')
        else:
            new_modes.append(mode)
    args.modes = new_modes
    return args

def suffixify(name, suffix, not_json):
    return name[:(p:=name.rindex('.') if '.' in name else len(name))] + suffix + (name[p:] if not_json else '.json')

def eval_parse(fname, args):
    # Set up as JSON where each top level is the general attempt string
    # and it is followed by an array of score values per epoch
    with open(suffixify(fname, args.suffix, args.not_json), 'w') as outfile:
        with open(fname, 'r') as infile:
            jdict = {}
            for idx, line in enumerate(infile.readlines()):
                try:
                    run_id = line[line.rindex('/')+1:line.index('epoch_')]
                    epoch = int(line[line.index('epoch_')+len('epoch_'):line.index('.pt')])
                    score = float(line[line.index("'score': ")+len("'score': "):line.index(", 'counts':")])
                except ValueError as e:
                    #print(f"{e.__class__.__name__} on line {idx}")
                    #if hasattr(e, 'msg'):
                    #    print(e.msg)
                    #if hasattr(e, 'message'):
                    #    print(e.message)
                    #print(line)
                    continue
                if run_id in jdict.keys():
                    # Epochs are 1-based, indices are 0-based
                    if (run_len := len(jdict[run_id])) == epoch - 1:
                        jdict[run_id].append(score)
                    elif run_len > epoch:
                        jdict[run_id][epoch-1] = score
                    else:
                        jdict[run_id].extend([None] * (epoch-1 - run_len))
                        jdict[run_id].append(score)
                else:
                    if epoch == 1:
                        jdict[run_id] = [score]
                    else:
                        jdict[run_id] = [None] * (epoch-1)
                        jdict[run_id].append(score)
            print(jdict)
            json.dump(jdict, outfile)

def train_parse(fname, args):
    # Set up as JSON where each top level is the general attempt string
    # and it is followed by an array of score values per epoch
    with open(suffixify(fname, args.suffix, args.not_json), 'w') as outfile:
        with open(fname, 'r') as infile:
            jdict = {}
            dummy_eval = 2
            final_iter = None
            training_flag = False
            skip_count = 0
            epoch = 0
            for idx, line in enumerate(infile.readlines()):
                if line == '\n':
                    continue
                if final_iter is None:
                    if 'Epoch' in line:
                        if dummy_eval > 0:
                            dummy_eval -= 1
                        else:
                            final_iter = int(line[line.index('/')+1:line.index('[')])-1
                            dummy_eval = 1
                    else:
                        continue
                else:
                    if not training_flag:
                        try:
                            start_idx = line.index('| ')
                            this_iter = int(line[start_idx+len('| '):line.index('/')])
                        except ValueError:
                            # Hit first pipe instead of second one
                            start_idx = line.index('| ', start_idx+len('| '))
                            this_iter = int(line[start_idx+len('| '):line.index('/')])
                        if this_iter != final_iter:
                            continue
                        training_flag = True
                        # Get epoch time, skip count = broken + too_big, 
                        epoch_time = line[line.index('[')+1:line.index('<')]
                        broken = int(line[line.index('broken=')+len('broken='):line.index(', max')])
                        too_big = int(line[line.index('too_big=')+len('too_big='):line.index(']')])
                        skip_count = skip_count + broken + too_big
                    else:
                        # Get epoch #, loss
                        if not line.startswith('Training'):
                            continue
                        if dummy_eval > 0:
                            dummy_eval -= 1
                            continue
                        dummy_eval = 2
                        epoch = int(line[line.index('| ')+len('| '):line.index('/')])
                        loss = float(line[line.index('loss=')+len('loss='):line.index(']')])
                        # Create entry
                        jdict[epoch] = {'time': epoch_time, 'seen': final_iter + 1 - skip_count, 'loss': loss}
                        training_flag = False
                        final_iter = None
            print(jdict)
            json.dump(jdict, outfile)

def main(args):
    print(args)
    parsed_warning = False
    for fname, mode in zip(args.files, args.modes):
        if not args.recompute and os.path.exists(suffixify(fname, args.suffix, args.not_json)):
            print(f"Parsed output '{suffixify(fname, args.suffix, args.not_json)}' already exists for {fname}. Skip.")
            if not parsed_warning:
                parsed_warning = True
                print(f"Use --recompute to force recomputation of files with pre-parsed outputs")
            continue
        if mode == 'eval':
            eval_parse(fname, args)
        else:
            train_parse(fname, args)

if __name__ == '__main__':
    main(parse(build()))

