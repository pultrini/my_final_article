import argparse
from experiment import exec_once
import torch

DATASET = 'PathMNIST'
NUM_EPOCHS = 10
OUTPUT_FILE = 'results/metrics/metrics_model1.json'
# Corrige o device: use 'cuda' ou 'cpu'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='normal', choices=['normal', 'complexity', 'min_loss'])
    parser.add_argument('--round', dest='round_id', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    # Semente opcional
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # IMPORTANTE: garanta que exec_once seja chamado mesmo quando seed não for passada
    exec_once(
        dataset_name=DATASET,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        output_file_path=OUTPUT_FILE,
        strategy=args.strategy
    )