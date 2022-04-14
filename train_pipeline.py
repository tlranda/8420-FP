import torch
import datasets
from evaluation_pipeline import (
    load_wmt14,
    evaluate,
)

def main():
    # Select device
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    if device == 'cpu':
        print(f"Warning! Running on CPU!")
    # Fetch dataset
    data = load_wmt14()
    # Create model
    # Train
    # Final evaluation
    trained_translator = lambda de: de
    print(evaluate(data, trained_translator))

if __name__ == '__main__':
    main()

