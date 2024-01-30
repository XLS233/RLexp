import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

a = [1, 2, 3]
b = [3, 4, 5]
c = [6, 7, 8]
a = torch.tensor(a)
b = torch.tensor(b).reshape(-1, 1)
c = torch.tensor(c)

array = (a, b, c)
dataset = da
