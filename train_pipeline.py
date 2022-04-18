import torch
import datasets
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
    prs.add_argument('--nhid', type=int, default=256, help='Hidden units per layer')
    prs.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    prs.add_argument('--nhead', type=int, default=2, help='Number of heads in encoder/decoder blocks')
    prs.add_argument('--lr', type=float, default=20, help='Initial learning rate')
    prs.add_argument('--clip', type=float, default=0.25, help='Gradient Clipping')
    prs.add_argument('--epochs', type=int, default=40, help='Upper epoch limit')
    prs.add_argument('--batch_size', type=int, default=20, help='Batch size')
    prs.add_argument('--bptt', type=int, default=35, help='Sequence length')
    prs.add_argument('--dropout', type=float, default=0.2, help='Dropout per layer (0 = no dropout)')
    prs.add_argument('--seed', type=int, default=2022, help='RNG Seed')
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
    # Train
    for epoch in range(1, args.epochs+1):
        transformer.train(data['train'])
        # Validate
    # Final evaluation
    trained_translator = lambda de: transformer.evaluate(de)
    print(evaluate(data['test'], trained_translator))

if __name__ == '__main__':
    main(parse(build()))

