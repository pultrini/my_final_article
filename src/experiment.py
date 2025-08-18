import os
import json

import torch
import torch.nn as nn
import numpy as np
import mlflow

from model import create_model
from data import data_transform
from functions import train, validate

def exec_once(dataset_name: str, num_epochs: int, device: str, output_file_path: str, strategy: str = "normal", type: str = None):
    """
    Executa uma única rodada de treinamento/validação, encontra as melhores
    métricas e salva o resultado em um arquivo JSON. Também registra tudo no MLflow.

    strategy: "normal" | "complexity" | "min_loss"
    """
    print(f"--- Iniciando rodada para dataset: {dataset_name} ---")

    # Validação básica da estratégia
    valid_strategies = {"normal", "complexity", "min_loss"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy inválida: {strategy}. Use uma de {sorted(valid_strategies)}")

    # Define/seleciona o experimento no MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_id=175611549495715788)

    # IMPORTANTE: use run_name para criar um novo run (não passe string posicional)
    with mlflow.start_run(run_name=f"{dataset_name}_{strategy}_run"):
        # Parâmetros do experimento
        mlflow.log_params({
            "dataset": dataset_name,
            "num_epochs": num_epochs,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "strategy": strategy,
        })

        # Dataloaders
        train_loader, val_loader, _ = data_transform(dataset=dataset_name)

        # Modelo/otimizador/loss
        model = create_model(dataset=dataset_name, type=type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Histórico por época
        hist_acc, hist_loss, hist_complexity = [], [], []

        for epoch in range(1, num_epochs + 1):
            # train retorna: epoch_loss_train, epoch_acc_train, entropy, disequilibrium, complexity
            _, _, _, _, complexity = train(
                model, train_loader, criterion, optimizer, device, dataset_name=dataset_name
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"Epoch {epoch}/{num_epochs} -> Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

            hist_acc.append(val_acc)
            hist_loss.append(val_loss)
            hist_complexity.append(complexity)
            best_loss = 99999.9
            best_complexity = -9999.9
            # Log por época no MLflow
            mlflow.log_metric("val_accuracy", float(val_acc), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            mlflow.log_metric("complexity", float(complexity), step=epoch)
            if val_loss < best_loss:
                torch.save(model.state_dict(), '/home/users/u12559743/Documentos/my_final_article/models/min_loss.pth')
            if complexity > best_complexity:
                torch.save(model.state_dict(), '/home/users/u12559743/Documentos/my_final_article/models/max_complexity.pth')
                

        # Agregados finais
        best_accuracy = float(np.max(hist_acc))
        min_loss = float(np.min(hist_loss))
        max_complexity_val = float(np.max(hist_complexity))

        # Épocas (base 1)
        epoch_max_complexity = int(np.argmax(hist_complexity)) + 1
        epoch_min_loss = int(np.argmin(hist_loss)) + 1

        # Log final no MLflow
        mlflow.log_metrics({
            "best_accuracy": best_accuracy,
            "min_loss": min_loss,
            "max_complexity": max_complexity_val,
        })
        mlflow.log_params({
            "epoch_max_complexity": epoch_max_complexity,
            "epoch_min_loss": epoch_min_loss,
        })

        # Salva JSON de saída
        resultados = {
            "Max_accuracy": best_accuracy,
            "Min_loss": min_loss,
            "Max_complexity": max_complexity_val,
        }

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump(resultados, f, indent=4)

        # Opcional: guarda o JSON como artifact do run
        mlflow.log_artifact(output_file_path)

        print(f"Resultados da rodada salvos em: {output_file_path}")
