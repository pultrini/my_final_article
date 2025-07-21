import os
import torch
import torch.nn as nn
import numpy as np
import json

from model import create_model
from data import data_transform
from functions import train, validate

def exec_once(dataset_name: str, num_epochs: int, device: str, output_file_path: str):
    """
    Executa uma única rodada de treinamento/validação, encontra as melhores
    métricas e salva o resultado em um arquivo JSON.
    """
    print(f"--- Iniciando rodada para dataset: {dataset_name} ---")

    train_loader, val_loader, _ = data_transform(dataset=dataset_name)
    
    model = create_model(dataset=dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    hist_acc, hist_loss, hist_complexity = [], [], []

    for epoch in range(1, num_epochs + 1):
        _, _, _, _, complexity = train(model, train_loader, criterion, optimizer, device, dataset_name= dataset_name)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs} -> Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

        hist_acc.append(val_acc)
        hist_loss.append(val_loss)
        hist_complexity.append(complexity)

    melhor_acc = max(hist_acc)
    menor_loss = min(hist_loss)
    idx_melhor_acc = hist_acc.index(melhor_acc)
    complexidade_final = hist_complexity[idx_melhor_acc]
    
    resultados = {
        "Max_accuracy": melhor_acc,
        "Min_loss": menor_loss,
        "Max_complexity": complexidade_final
    }

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(resultados, f, indent=4)
        
    print(f"Resultados da rodada salvos em: {output_file_path}")