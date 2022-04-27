import torch
import datasets
import tqdm
import argparse
from evaluation_pipeline import (
    load_wmt14,
    evaluate,
)
import models

# Define command line interface
def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--model', type=str, choices=models.choices, default=models.choices[0], help='Model to train')
    # Hyperparameters
    prs.add_argument('--seed', type=int, default=2022, help='RNG Seed')
    prs.add_argument('--nhid', type=int, default=None, help='Hidden units per layer (Leave unspecified to use pretrained BERT embeddings)')
    prs.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    prs.add_argument('--nhead', type=int, default=2, help='Number of heads in encoder/decoder blocks')
    prs.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    prs.add_argument('--dropout', type=float, default=0.2, help='Dropout per layer (0 = no dropout)')
    # Iteration Controls
    prs.add_argument('--epochs', type=int, default=40, help='Upper epoch limit')
    prs.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    prs.add_argument('--eval-size', type=int, default=10, help='Evaluation batch size (usually should be significantly lower than train)')
    prs.add_argument('--eval-limit', type=int, default=None, help='Maximum number of examples to evaluate (default all)')
    prs.add_argument('--train-limit', type=int, default=None, help='Maximum number of examples to train each epoch (default all)')
    # Persistence
    prs.add_argument('--save', type=str, default=None, help='Path and prefix to save epoch results to (default None)')
    prs.add_argument('--load', type=str, default=None, help='Path and prefix to load partial results from (default None)')
    prs.add_argument('--skip-load', type=str, default=None, help='Path to load serialized batch ids to skip from (default None)')
    prs.add_argument('--skip-save', type=str, default=None, help='Path to save serialized batch ids to skip to (default None)')
    return prs

# Any postprocessing to validate or prepare arguments
def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def main(args):
    print(args)
    # Select device
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    if device == 'cpu':
        print(f"Warning! Running on CPU!")
    # Fetch dataset
    data = load_wmt14()
    # Create model
    transformer = models.lookup[args.model](args)
    if args.load is None:
        epoch_start = 1
    else:
        # Load state dict
        transformer_params = torch.load(args.load)
        transformer.load_state_dict(transformer_params)
        # Saved as prefix..epoch_######.pt
        epoch_start = int(args.load[args.load.index('epoch_')+len('epoch_'):-3])
    # Train
    epoch_bar = tqdm.auto.tqdm(range(epoch_start, args.epochs+1), desc="Training: ", unit='epoch', leave=False)
    for epoch in range(epoch_start, args.epochs+1):
        loss = transformer.train(data['train'], limit=args.train_limit)
        # Save
        if args.save is not None:
            torch.save(transformer.state_dict(), args.save+'epoch_'+str(epoch)+'.pt')
        # Validation not performed to save time
        epoch_bar.set_postfix(loss=loss)
        epoch_bar.update(1)
    # Serialize skippable batches
    if args.skip_save is not None:
        torch.save(transformer.skip_batch, args.skip_save)
    # Final evaluation
    trained_translator = lambda de: transformer.evaluate(de)
    print(evaluate(data['test'], trained_translator, batch_size=args.eval_size, limit=args.eval_limit))

if __name__ == '__main__':
    main(parse(build()))

