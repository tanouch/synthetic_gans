from my_imports import *

from defining_new_activations_and_layers import BjorckLinear, MaxMin
from tools import convert_to_gpu
torch.manual_seed(5015)

class Importance_weighter(nn.Module):
    def __init__(self, spectral_normalization = False):
        super(Importance_weighter, self).__init__()
        self.linear = nn.Sequential(nn.Linear(2, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 1))

    def forward(self, x):
        w = self.linear(x)
        return w


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        if config.activation_function=="GeLU":
            activation = nn.GELU()
        elif config.activation_function=="ReLU":
            activation = nn.ReLU()
        array = [nn.Linear(config.z_dim, config.g_width), activation]
        array += [nn.Linear(config.g_width, config.g_width), activation]*(config.g_depth-1)
        array += [nn.Linear(config.g_width, config.output_dim)]
        self.linear = nn.Sequential(*array)

    def forward(self, x):
        return self.linear(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        array = [nn.Linear(config.output_dim, config.d_width), MaxMin(num_units=config.d_width/2)]
        array += [nn.Linear(config.d_width, config.d_width), MaxMin(num_units=config.d_width/2)]*(config.d_depth-1)
        array += [nn.Linear(config.d_width, 1)]
        self.linear = nn.Sequential(*array)

    def forward(self, x):
        return self.linear(x)

class Discriminator_bjorckReLU(nn.Module):
    def __init__(self, config, cuda=False):
        super(Discriminator_bjorckReLU, self).__init__()
        config_bjorck = Munch.fromDict(dict(cuda=cuda, model=dict(linear=dict(bjorck_beta=0.5, bjorck_iter=4, bjorck_order=1, safe_scaling=True))))
        array = [BjorckLinear(config.output_dim, config.d_width, config=config_bjorck), nn.ReLU()]
        array += [BjorckLinear(config.d_width, config.d_width, config=config_bjorck), nn.ReLU()]*(config.d_depth-1)
        array += [BjorckLinear(config.d_width, 1, config=config_bjorck)]
        self.linear = nn.Sequential(*array)

    def forward(self, x):
        return self.linear(x)

class Discriminator_bjorckGroupSort(nn.Module):
    def __init__(self, config, cuda=False):
        super(Discriminator_bjorckGroupSort, self).__init__()
        config_bjorck = Munch.fromDict(dict(cuda=cuda, model=dict(linear=dict(bjorck_beta=0.5, bjorck_iter=3, bjorck_order=1, safe_scaling=True))))
        array = [BjorckLinear(config.output_dim, config.d_width, config=config_bjorck), MaxMin(num_units=config.d_width/2)]
        array += [BjorckLinear(config.d_width, config.d_width, config=config_bjorck), MaxMin(num_units=config.d_width/2)]*(config.d_depth-1)
        array += [BjorckLinear(config.d_width, 1, config=config_bjorck)]
        self.linear = nn.Sequential(*array)

    def forward(self, x):
        #return self.linear(x)
        return torch.clamp(self.linear(x), min=-100., max=100.0)

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, config = None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = convert_to_gpu(alpha, config)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)

        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=convert_to_gpu(torch.ones(disc_interpolates.size()), config),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def cal_gradient_penalty_generator(netG, gz, z, constant=10.0, config = None):
    gradients = torch.autograd.grad(outputs=gz, inputs=z, grad_outputs=convert_to_gpu(torch.ones(gz.size()), config),
                                        create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(gz.size(0), -1)
    gradient_penalty = (torch.clamp(((gradients + 1e-16).norm(2, dim=1) - constant), max=0.0) ** 2).mean()
    return gradient_penalty


class Classifier_mnist(nn.Module):
    def __init__(self, config):
        super(Classifier_mnist, self).__init__()
        if config.dataset == 'cifar10':
            input_nc = 3
            self.linear_1 = nn.Sequential(
                nn.Linear(64*8*8,128),
                nn.ReLU())
        else:
            input_nc = 1
            self.linear_1 = nn.Sequential(
                nn.Linear(64*7*7,128),
                nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=4, stride = 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=4, stride = 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.softmax = nn.Sequential(
            nn.Linear(128,10),
            nn.Softmax()
        )

    def forward(self, x):
        embedding = self.conv(x)
        embedding = embedding.view(x.shape[0], -1)
        embedding = self.linear_1(embedding)
        preds = self.softmax(embedding)
        return embedding, preds


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class Generator_mnist(nn.Module):
    def __init__(self, config, nc = 128):
        super(Generator_mnist, self).__init__()
        self.config = config
        self.nc = nc
        self.dense = nn.Sequential(
            nn.Linear(config.z_dim, 7*7*nc)
        )

        nc = nc//2
        self.conv_1 = nn.Sequential(
            nn.Conv2d(nc*2, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        nc = nc//2
        self.conv_2 = nn.Sequential(
            nn.Conv2d(nc*2, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        output_nc = 1
        self.conv_final = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc, output_nc, kernel_size=3, padding=1))

        self.interpol_1 = Interpolate(size = (14,14), mode = 'nearest')
        self.interpol_2 = Interpolate(size = (28,28), mode = 'nearest')
        self.final_activation = nn.Tanh()

    def forward(self, z):
        z = self.dense(z)
        img = z.reshape(z.shape[0],self.nc,7,7)
        img = self.conv_1(img)
        img = self.interpol_1(img)
        img = self.conv_2(img)
        img = self.interpol_2(img)
        img = self.conv_final(img)
        img = self.final_activation(img)
        img = img / 2 + 0.5
        return img


class Discriminator_mnist(nn.Module):
    def __init__(self, config):
        super(Discriminator_mnist, self).__init__()
        input_nc = 1
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride = 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(64*7*7,1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
