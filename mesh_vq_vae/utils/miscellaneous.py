import random
import numpy as np
import torch


def set_seed(seed=0, n_gpu=1):
    """
    Borrowed from FastMetro: https://github.com/postech-ami/FastMETRO/blob/main/src/utils/miscellaneous.py
    Set the seed for reproducing experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
