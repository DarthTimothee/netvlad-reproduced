import gc
import os
import psutil
import torch
from colorama import Fore
from tqdm import tqdm

process = psutil.Process(os.getpid())


def bar_fmt(color):
    return "{l_bar}%s{bar}%s{r_bar}" % (color, color)


def pbar(x=0, position=0, leave=True, desc='', color=Fore.RED, total=None, smoothing=0.3):
    if total:
        return tqdm(total=total, position=position, leave=leave, bar_format=bar_fmt(color), desc=f'{desc: <32}',
                    smoothing=smoothing)
    return tqdm(x, position=position, leave=leave, bar_format=bar_fmt(color), desc=f'{desc: <32}', smoothing=smoothing)


def ram_usage():
    return f"{process.memory_info().rss / 10 ** 9: .3} GB"


def current_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


def custom_distance(x1, x2):
    return torch.pairwise_distance(x1, x2) ** 2
