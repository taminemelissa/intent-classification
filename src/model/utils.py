import torch
from config.config import config

def get_device():
    return torch.device(f'cuda:{config.CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')