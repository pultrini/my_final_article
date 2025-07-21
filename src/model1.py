
from experiment import exec_once
import torch

DATASET = 'PathMNIST'
NUM_EPOCHS = 10
OUTPUT_FILE = 'results/metrics/metrics_model1.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    exec_once(
        dataset_name=DATASET,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        output_file_path=OUTPUT_FILE
    )