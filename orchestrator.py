import subprocess
import json
import os
import math
import shlex

def run_script(script_path, args=None):
    """
    Executa um script Python em subprocess, com argumentos de linha de comando.
    Retorna o stdout em caso de sucesso; None em caso de erro.
    """
    cmd = ['python3', script_path]
    if args:
        cmd += args

    print(f"Executando: {' '.join(shlex.quote(p) for p in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"--- Erro ao executar {script_path} ---\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n----")
        return None
    else:
        print(f">>> {script_path} executado com sucesso.")
        return result.stdout

def read_metrics(file_path):
    """Lê um arquivo de métricas JSON e retorna o dicionário ou None se não existir."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        print(f"Arquivo de métricas não encontrado: {file_path}")
        return None

MAX_ITERATIONS = 5
BASE_SEED = 1234
all_metrics = {}
METRICS_DIR = 'results/metrics'
AGGREGATED_METRICS_FILE = 'results/metrics/all_metrics.json'
MODELS = [
    {
        'name': 'model1_normal',
        'script': 'src/model1.py',
        'metrics_file': os.path.join(METRICS_DIR, 'metrics_model1.json'),
        'strategy': 'normal',
    },
    {
        'name': 'model2_complexity',
        'script': 'src/model2.py',
        'metrics_file': os.path.join(METRICS_DIR, 'metrics_model2.json'),
        'strategy': 'complexity',
    },
    {
        'name': 'model3_loss',
        'script': 'src/model3.py',
        'metrics_file': os.path.join(METRICS_DIR, 'metrics_model3.json'),
        'strategy': 'min_loss',
    },
]

print(f"Iniciando loop de {MAX_ITERATIONS} iterações...")

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n===== Iteração Estatística {iteration}/{MAX_ITERATIONS} =====")
    #seed = BASE_SEED + iteration
    for model in MODELS:
        args = [
            '--strategy', model['strategy'],
            '--round', str(iteration),
            #'--seed', str(seed)
        ]
        run_script(model['script'], args= args)
        metrics_per_model = {}
        for m in MODELS:
            metrics_per_model[m['name']] = read_metrics(m['metrics_file'])


    if all(metrics_per_model.values()):
        all_metrics[f'iteration_{iteration}'] = metrics_per_model
        with open(AGGREGATED_METRICS_FILE, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(">>> Métricas da iteração salvas em all_metrics.json")
    else:
        print("Algumas métricas não foram encontradas nesta iteração. Pulando agregação desta iteração.")

print("\n==== Loop finalizado. Calculando estatísticas finais... ====")

try:
    if not os.path.exists(AGGREGATED_METRICS_FILE):
        raise FileNotFoundError(f"O arquivo agregado '{AGGREGATED_METRICS_FILE}' não foi encontrado.")

    with open(AGGREGATED_METRICS_FILE, 'r') as file:
        data = json.load(file)

    model_keys = [m['name'] for m in MODELS]
    num_rounds = len(data)
    if num_rounds == 0:
        raise ValueError("Nenhuma rodada agregada encontrada em all_metrics.json.")


    sum_accuracy = {model: 0.0 for model in model_keys}
    sum_loss = {model: 0.0 for model in model_keys}
    sum_complexity = {model: 0.0 for model in model_keys}

    for round_data in data.values():
        for model in model_keys:
            sum_accuracy[model] += round_data[model].get('Max_accuracy', 0.0)
            sum_loss[model] += round_data[model].get('Min_loss', 0.0)
            sum_complexity[model] += round_data[model].get('Max_complexity', 0.0)

    avg_accuracy = {model: val / num_rounds for model, val in sum_accuracy.items()}
    avg_loss = {model: val / num_rounds for model, val in sum_loss.items()}
    avg_complexity = {model: val / num_rounds for model, val in sum_complexity.items()}


    sum_sq_accuracy = {model: 0.0 for model in model_keys}
    sum_sq_loss = {model: 0.0 for model in model_keys}
    sum_sq_complexity = {model: 0.0 for model in model_keys}

    for round_data in data.values():
        for model in model_keys:
            sum_sq_accuracy[model] += (round_data[model].get('Max_accuracy', 0.0) - avg_accuracy[model]) ** 2
            sum_sq_loss[model] += (round_data[model].get('Min_loss', 0.0) - avg_loss[model]) ** 2
            sum_sq_complexity[model] += (round_data[model].get('Max_complexity', 0.0) - avg_complexity[model]) ** 2

    denom = (num_rounds - 1) if num_rounds > 1 else 1
    std_dev_accuracy = {model: math.sqrt(val / denom) for model, val in sum_sq_accuracy.items()}
    std_dev_loss = {model: math.sqrt(val / denom) for model, val in sum_sq_loss.items()}
    std_dev_complexity = {model: math.sqrt(val / denom) for model, val in sum_sq_complexity.items()}

    final_results = {}
    for model in model_keys:
        final_results[model] = {
            'Avg_Max_Accuracy': f"{avg_accuracy[model]:.2f} ± {std_dev_accuracy[model]:.2f}",
            'Avg_Min_Loss': f"{avg_loss[model]:.4f} ± {std_dev_loss[model]:.4f}",
            'Avg_Max_Complexity': f"{avg_complexity[model]:.2f} ± {std_dev_complexity[model]:.2f}",
        }

    FINAL_RESULTS_FILE = 'results/results.json'
    os.makedirs(os.path.dirname(FINAL_RESULTS_FILE), exist_ok=True)
    with open(FINAL_RESULTS_FILE, 'w') as file:
        json.dump(final_results, file, indent=4)

    print("\n--- RESULTADOS FINAIS ---")
    print(json.dumps(final_results, indent=4))
    print(f"\nEstatísticas calculadas e salvas em {FINAL_RESULTS_FILE}!")

except FileNotFoundError as e:
    print(str(e))
except Exception as e:
    print(f"Ocorreu um erro inesperado durante o cálculo das estatísticas: {e}")