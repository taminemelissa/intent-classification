import torch
from config import config
from typing import List
import numpy as np


def get_device():
    return torch.device(f'cuda:{config.CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')


def print_array_stats(arr: List, metric: str = '', model_name: str = '', decimal: int = 1) -> str:
    if arr:
        result_line = f'Number of test samples: {len(arr)}' \
                      f'Model_name: {model_name}' \
                      f'Metric: {metric}' \
                      f'{round(float(np.mean(arr)), decimal)} ' \
                      f'(min: {round(float(np.min(arr)), decimal)}, ' \
                      f'max: {round(float(np.max(arr)), decimal)}, ' \
                      f'sd: {round(float(np.std(arr)), decimal)}, ' \
                      f'median: {round(float(np.median(arr)), decimal)})'
        print(result_line)
        return result_line
    else:
        print(f'{model_name}: NA')
        return f'{model_name}: NA'
