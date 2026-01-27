import torch
import torch.nn as nn

class MinMaxObserver:
    def __init__(self):
        self.rmin = None
        self.rmax = None

    def observe(self, x: torch.Tensor):
        x_min = x.min().item()
        x_max = x.max().item()

        if self.rmin is None:
            self.rmin = x_min
            self.rmax = x_max
        else:
            self.rmin = min(self.rmin, x_min)
            self.rmax = max(self.rmax, x_max)

    def get_minmax(self):
        return self.rmin, self.rmax
