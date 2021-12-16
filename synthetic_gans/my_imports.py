import sys, json, random, math, copy, os, functools, argparse
import numpy as np
import matplotlib.pyplot as plt
from time import time
from itertools import product
import ot

from scipy import ndimage,linalg
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.nn.utils import spectral_norm
from torch.distributions import MultivariateNormal, uniform, categorical
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch import Tensor