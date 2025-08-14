import argparse
from experiment import exec_once
import torch

DATASET = 'OrganAMNIST'
NUM_EPOCHS = 10
OUTPUT_FILE = 'results/metrics/metrics_model2.json'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='normal', choices=['normal', 'complexity', 'min_loss'])
    parser.add_argument('--round', dest='round_id', type=int)  # opcional, não usado aqui
    parser.add_argument('--seed', type=int)  # opcional
    args = parser.parse_args()

    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    exec_once(
        dataset_name=DATASET,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        output_file_path=OUTPUT_FILE,
        strategy=args.strategy
    )