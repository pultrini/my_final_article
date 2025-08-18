import torch

   # Verifica se CUDA está disponível
if torch.cuda.is_available():
       num_gpus = torch.cuda.device_count()
       print(f"Número de GPUs disponíveis: {num_gpus}")
       
       for i in range(num_gpus):
           gpu_name = torch.cuda.get_device_name(i)
           print(f"GPU {i}: {gpu_name}")
           print(torch.cuda.get_device_properties(i))
else:
       print("Nenhuma GPU com CUDA disponível. Verifique sua instalação ou hardware.")