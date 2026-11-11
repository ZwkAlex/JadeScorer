import os
from pathlib import Path
from typing import Union

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn
import csv


def load_yaml(hyp_path: Union[Path, str]) -> dict:
    if isinstance(hyp_path, str) or isinstance(hyp_path, Path):
        with open(str(hyp_path), errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print(colorstr(f'{Path(hyp_path).name} hyper-parameters: ')
          + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    return hyp


def colorstr(*input_str):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input_str if len(input_str) > 1 else (
        'blue', 'bold', input_str[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def select_device(device: str = '', batch_size: int = 0) -> torch.device:
    s = f'select device \'{device}\' in args \t' if device != '' else ''
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        p = torch.cuda.get_device_properties(0)
        s += f"CUDA:{device} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(colorstr('device select: ') + s.strip('\n'))
    return torch.device(f'cuda:0' if cuda else 'cpu')


class AvgLoss:
    def __init__(self):
        self.loss = 0
        self.count = 0

    def update(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self.loss += loss
        self.count += 1

    def average(self):
        return self.loss / self.count


class BestSaver:
    def __init__(self):
        self.best_result = float("inf")
        self.current_result = None

    def save(self, result):
        self.current_result = result
        self.best_result = min(result, self.best_result)
        print(f"Best:", colorstr("red", "bold", f"{self.best_result:.2f}"), f"\tCurrent:",
              colorstr("red", f"{self.current_result: .2f}"))


class ResultSaver:
    def __init__(self, model_name):
        base_dir = Path("./results")
        os.makedirs(base_dir, exist_ok=True)
        self.writer = csv.writer(open(base_dir / f"{model_name}.csv", "w"))
        self.writer.writerow(["Epoch", "Color_MAE", "Pixel_MAE"])
        self.color_saver = BestSaver()
        self.pixel_mae = BestSaver()

    def save(self, epoch, color_mae, pixel_mae):
        color_mae = color_mae.item() if isinstance(color_mae, torch.Tensor) else color_mae
        pixel_mae = pixel_mae.item() if isinstance(pixel_mae, torch.Tensor) else pixel_mae
        self.writer.writerow([epoch, color_mae, pixel_mae])
        self.color_saver.save(color_mae)
        self.pixel_mae.save(pixel_mae)


class CustomFigure:
    def __init__(self, title, dpi, alpha):
        self.title = title
        self.dpi = dpi
        self.alpha = alpha
        self.fig = plt.figure(dpi=dpi)
        self.ax = plt.subplot(111)
        self.fig_list = list()

    def reset(self):
        self.fig.clear()
        plt.close(self.fig)
        self.fig = plt.figure(dpi=self.dpi)
        self.ax = plt.subplot(111)

    @staticmethod
    def type_unify(x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        return x

    def get_figure(self):
        self.ax.legend()
        self.ax.set_title(self.title)
        return self.fig


class HistFigure(CustomFigure):
    def __init__(self, title, sample_num=5000, dpi=200, alpha=0.6):
        super().__init__(title, dpi, alpha)
        self.title = title
        self.sample_num = sample_num

    def add_data(self, x, legend="data"):
        x = self.type_unify(x)
        if len(x) > self.sample_num:
            x = np.random.choice(x, self.sample_num, replace=False)
        self.ax.hist(x, alpha=self.alpha, label=legend, rwidth=0.5, density=True)
        self.ax.set_xlabel("Score")
        self.ax.set_ylabel("Density")
        self.ax.set_xlim(1, 10)
        self.ax.set_xticks(np.linspace(1, 10, 10))

    def save(self, name):
        plt.savefig(f'./{name}.jpg', dpi=500, bbox_inches='tight')
