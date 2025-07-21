import subprocess
import json
import os
import math

def run_script(script_path):
    
    print(f"Executando: python -m {script_path}")
    result = subprocess.run(
        ['python3', script_path], 
        capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(f'--- Erro ao executar {script_path} ---\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n--------------------')
        return None
    else:
        print(f'>>> {script_path} executado com sucesso.')
        return result.stdout

def read_metrics(file_path):
    """Lê um arquivo de métricas JSON."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        print(f"Arquivo de métricas não encontrado: {file_path}")
        return None

MAX_ITERATIONS = 5 
all_metrics = {}
AGGREGATED_METRICS_FILE = 'results/metrics/all_metrics.json'

print(f"Iniciando loop de {MAX_ITERATIONS} iterações...")

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n===== Iteração Estatística {iteration}/{MAX_ITERATIONS} =====")
    
    run_script('src/model1.py')
    run_script('src/model2.py')
    run_script('src/model3.py')

    metrics1 = read_metrics('results/metrics/metrics_model1.json')
    metrics2 = read_metrics('results/metrics/metrics_model2.json')
    metrics3 = read_metrics('results/metrics/metrics_model3.json')
    
    if all((metrics1, metrics2, metrics3)):
        all_metrics[f'iteration_{iteration}'] = {
            'model1_normal': metrics1,
            'model2_complexity': metrics2,
            'model3_loss': metrics3
        }
        os.makedirs(os.path.dirname(AGGREGATED_METRICS_FILE), exist_ok=True)
        with open(AGGREGATED_METRICS_FILE, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(">>> Métricas da iteração salvas em all_metrics.json")

print("\n===== Loop finalizado. Calculando estatísticas finais... =====")

try:
    with open(AGGREGATED_METRICS_FILE, 'r') as file:
        data = json.load(file)

    model_keys = ['model1_normal', 'model2_complexity', 'model3_loss']
    num_rounds = len(data)

    sum_accuracy = {model: 0 for model in model_keys}
    sum_loss = {model: 0 for model in model_keys}
    sum_complexity = {model: 0 for model in model_keys}

    for round_data in data.values():
        for model in model_keys:
            sum_accuracy[model] += round_data[model]['Max_accuracy']
            sum_loss[model] += round_data[model]['Min_loss']
            sum_complexity[model] += round_data[model]['Max_complexity']

    avg_accuracy = {model: val / num_rounds for model, val in sum_accuracy.items()}
    avg_loss = {model: val / num_rounds for model, val in sum_loss.items()}
    avg_complexity = {model: val / num_rounds for model, val in sum_complexity.items()}

    sum_sq_accuracy = {model: 0 for model in model_keys}
    sum_sq_loss = {model: 0 for model in model_keys}
    sum_sq_complexity = {model: 0 for model in model_keys}

    for round_data in data.values():
        for model in model_keys:
            sum_sq_accuracy[model] += (round_data[model].get('Max_accuracy', 0) - avg_accuracy[model])**2
            sum_sq_loss[model] += (round_data[model].get('Min_loss', 0) - avg_loss[model])**2
            sum_sq_complexity[model] += (round_data[model].get('Max_complexity', 0) - avg_complexity[model])**2

    std_dev_accuracy = {model: math.sqrt(val / num_rounds) for model, val in sum_sq_accuracy.items()}
    std_dev_loss = {model: math.sqrt(val / num_rounds) for model, val in sum_sq_loss.items()}
    std_dev_complexity = {model: math.sqrt(val / num_rounds) for model, val in sum_sq_complexity.items()}

    final_results = {}
    for model in model_keys:
        final_results[model] = {
            'Avg_Max_Accuracy': f"{avg_accuracy[model]:.2f} ± {std_dev_accuracy[model]:.2f}",
            'Avg_Min_Loss': f"{avg_loss[model]:.4f} ± {std_dev_loss[model]:.4f}",
            'Avg_Max_Complexity': f"{avg_complexity[model]:.2f} ± {std_dev_complexity[model]:.2f}"
        }
    
    FINAL_RESULTS_FILE = 'results/results.json'
    with open(FINAL_RESULTS_FILE, 'w') as file:
        json.dump(final_results, file, indent=4)

    print("\n--- RESULTADOS FINAIS ---")
    print(json.dumps(final_results, indent=4))
    print(f"\nEstatísticas calculadas e salvas em {FINAL_RESULTS_FILE}!")

except FileNotFoundError:
    print(f"Erro: O arquivo agregado '{AGGREGATED_METRICS_FILE}' não foi encontrado. Verifique se as execuções dos modelos ocorreram sem erros.")
except Exception as e:
    print(f"Ocorreu um erro inesperado durante o cálculo das estatísticas: {e}")