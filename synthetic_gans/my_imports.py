import sys, json, random, math, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os, functools, argparse
from time import time
from sklearn.manifold import TSNE
from sklearn import manifold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose
from itertools import product

import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from sklearn import manifold
from scipy import ndimage

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter
from torchvision import datasets
import torchvision.transforms as transforms
from torch.distributions import MultivariateNormal

from munch import Munch