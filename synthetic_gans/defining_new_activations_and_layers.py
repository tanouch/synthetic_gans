from my_imports import *
from maths.projections import *
from maths.projections import bjorck_orthonormalize, get_safe_bjorck_scaling
from tools import convert_to_gpu


class DenseLinear(nn.Module):
    def __init__(self):
        super(DenseLinear, self).__init__()

    def _set_network_parameters(self, in_features, out_features, bias=True, cuda=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _set_config(self, config):
        self.config = config

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        raise NotImplementedError

    def project_weights(self, proj_config):
        with torch.no_grad():
            projected_weights = project_weights(self.weight, proj_config, self.config.cuda)
            # Override the previous weights.
            self.weight.data.copy_(projected_weights)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

class BjorckLinear(DenseLinear):
    def __init__(self, in_features=1, out_features=1, bias=True, config=None):
        super(BjorckLinear, self).__init__()
        self._set_config(config)
        self._set_network_parameters(in_features, out_features, bias, cuda=config.cuda)

    def forward(self, x):
        # Scale the values of the matrix to make sure the singular values are less than or equal to 1.
        if self.config.model.linear.safe_scaling:
            scaling = get_safe_bjorck_scaling(self.weight, cuda=self.config.cuda)
        else:
            scaling = 1.0

        ortho_w = bjorck_orthonormalize(self.weight.t() / scaling,
                    beta=self.config.model.linear.bjorck_beta,
                    iters=self.config.model.linear.bjorck_iter,
                    order=self.config.model.linear.bjorck_order).t()
        return F.linear(x, ortho_w, self.bias)


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
    def forward(self, x):
        raise NotImplementedError

class MaxMin(Activation):
    def __init__(self, num_units, axis=-1):
        super(MaxMin, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        maxes = maxout(x, self.num_units, self.axis)
        mins = minout(x, self.num_units, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)

def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [int(num_channels // num_units)]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size

def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]

def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]