import os
import json
import random
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_hyperparams(model_name, params):
    os.makedirs('results', exist_ok=True)
    with open(f'results/hyperparams_{model_name}.json', 'w') as f:
        json.dump(params, f, indent=4)

def setup_environment():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    with open('results/env_info.txt', 'w') as f:
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available:  {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Model:       {torch.cuda.get_device_name(0)}\n")
            f.write(f"VRAM (MB):       {torch.cuda.get_device_properties(0).total_memory // 1024**2}\n")

    print(f"[Setup] Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")