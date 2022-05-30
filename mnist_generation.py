import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm

from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

import argparse
from einops import rearrange
#from stylegan2_pytorch import StyleGAN2

class Discriminator_style(nn.Module):
    def __init__(self, spectral_normalization):
        super(Discriminator_style, self).__init__()
        input_nc = 3
        hidden_nc = 512
        if spectral_normalization:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride = 2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(hidden_nc, hidden_nc, kernel_size=4, stride = 2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride = 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=4, stride = 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        if spectral_normalization:
            self.linear = nn.Sequential(spectral_norm(nn.Linear(hidden_nc*8*8,1,bias=False)))
        else:
            self.linear = nn.Sequential(nn.Linear(hidden_nc*8*8,1,bias=False))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class Mapping_style(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, depth = 1, linear = 0):
        super(Mapping_style, self).__init__()
        self.nc = nc
        layers = []
        for d in range(depth-1):
            layers.append(nn.Linear(z_dim,z_dim))
            if linear==0:
                layers.append(nn.LeakyReLU(0.2,inplace=True))
        self.dense = nn.Sequential(*layers)

    def forward(self, z):
        z = self.dense(z)
        return z

class StyleInstanceNorm2D(nn.Module):
    def __init__(self,planes,z_dim=128):
        super(StyleInstanceNorm2D,self).__init__()
        self.bn = nn.InstanceNorm2d(planes,affine = False)
        self.gamma = nn.Sequential(nn.Linear(z_dim,planes))
        self.beta = nn.Sequential(nn.Linear(z_dim,planes))

    def forward(self, x, z):
        gamma_z = self.gamma(z).unsqueeze(2).unsqueeze(3)
        beta_z = self.beta(z).unsqueeze(2).unsqueeze(3)
        return gamma_z * self.bn(x) + beta_z

class Style_block(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, tanh=1):
        super(Style_block, self).__init__()
        self.nc = nc
        nc = nc
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.in_norm = StyleInstanceNorm2D(nc,z_dim)

    def forward(self,x,z):
        x = self.conv(x)
        x = self.in_norm(x,z)
        return x

class Synthesis_style(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, tanh=1):
        super(Synthesis_style, self).__init__()
        self.nc = nc
        self.x = torch.nn.Parameter(torch.zeros(nc,8,8))
        self.conv_1 = Style_block(z_dim,nc)
        self.conv_1_1 = Style_block(z_dim,nc)
        self.conv_2 = Style_block(z_dim,nc)
        self.conv_2_1 = Style_block(z_dim,nc)
        self.conv_3 = Style_block(z_dim,nc)

        self.conv_final = nn.Sequential(
            nn.Conv2d(nc, 3, kernel_size=3, padding=1))

        self.interpol_1 = Interpolate(size = (16,16), mode = 'nearest')
        self.interpol_2 = Interpolate(size = (32,32), mode = 'nearest')
        self.tanh = tanh
        self.final_activation = nn.Tanh()

    def forward(self, z):
        img = self.x.unsqueeze(0).repeat(z.shape[0],1,1,1)
        img = self.conv_1(img,z)
        img = self.conv_1_1(img,z)
        img = self.interpol_1(img)
        img = self.conv_2(img,z)
        img = self.conv_2_1(img,z)
        img = self.interpol_2(img)
        img = self.conv_3(img,z)
        img = self.conv_final(img)
        if self.tanh==1:
            img = self.final_activation(img)
            img = img / 2 + 0.5
        return img

class Generator_style(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, depth = 1, tanh = 1, linear = 0):
        super(Generator_style, self).__init__()
        self.nc = nc
        self.mapping = Mapping_style(z_dim,nc,depth,linear)
        self.synthesis = Synthesis_style(z_dim,nc,tanh)

    def forward(self, z):
        z = self.mapping(z)
        img = self.synthesis(z)
        return img

class Generator_resnet(nn.Module):
    def __init__(self, nc = 128, s_norm_G = False, z_dim=128, tanh=1):
        super(Generator_resnet, self).__init__()
        self.nc = nc
        s_norm_G = s_norm_G
        if s_norm_G:
            self.dense = nn.Sequential(spectral_norm(nn.Linear(z_dim, 4*4*nc)))
        else:
            self.dense = nn.Sequential(nn.Linear(z_dim, 4*4*nc))

        self.resblock_1 = ResBlock_generator(nc,nc, act_function=nn.ReLU(), norm=True, snorm=s_norm_G,z_dim=z_dim)
        self.resblock_2 = ResBlock_generator(nc,nc, act_function=nn.ReLU(), norm=True, snorm=s_norm_G,z_dim=z_dim)
        self.resblock_3 = ResBlock_generator(nc,nc, act_function=nn.ReLU(), norm=True, snorm=s_norm_G,z_dim=z_dim)

        output_nc = 3
        if s_norm_G:
            self.conv_final = nn.Sequential(
                    spectral_norm(nn.Conv2d(nc, output_nc, kernel_size=3, padding=1)),
                    nn.Tanh())
        else:
            if tanh==1:
                self.conv_final = nn.Sequential(
                    nn.Conv2d(nc, output_nc, kernel_size=3, padding=1),
                    nn.Tanh())
            elif tanh==0:
                self.conv_final = nn.Sequential(
                    nn.Conv2d(nc, output_nc, kernel_size=3, padding=1))

        self.interpol_1 = Interpolate(size = (8,8), mode = 'nearest')
        self.interpol_2 = Interpolate(size = (16,16), mode = 'nearest')
        self.interpol_3 = Interpolate(size = (32,32), mode = 'nearest')


    def forward(self, z):
        img = self.dense(z)
        img = img.reshape(z.shape[0],self.nc,4,4)
        img = self.interpol_1(self.resblock_1(img,z))
        img = self.interpol_2(self.resblock_2(img,z))
        img = self.interpol_3(self.resblock_3(img,z))
        img = self.conv_final(img)
        return img

class Discriminator_resnet(nn.Module):
    def __init__(self, snorm=False, config=None):
        super(Discriminator_resnet, self).__init__()
        input_nc = 3
        self.config = config
        snorm = snorm
        nc = 256
        self.resblock_1 = ResBlock(3,nc, act_function=nn.ReLU(), norm=False, snorm = snorm)
        self.resblock_2 = ResBlock(nc,nc, act_function=nn.ReLU(), norm=False, snorm = snorm)
        self.resblock_3 = ResBlock(nc,nc, act_function=nn.ReLU(), norm=False, snorm = snorm)
        self.resblock_4 = ResBlock(nc,nc, act_function=nn.ReLU(), norm=False, snorm = snorm)
        self.down_pool = nn.AvgPool2d((2,2))
        if snorm:
            self.linear = nn.Sequential(spectral_norm(nn.Linear(nc,1)))
        else:
            self.linear = nn.Sequential(nn.Linear(nc,1))

    def forward(self, x, labels):

        x = self.down_pool(self.resblock_1(x))
        x = self.down_pool(self.resblock_2(x))
        x = self.resblock_4(self.resblock_3(x))
        x = (torch.sum(torch.sum(x, dim = 3), dim = 2))/(8*8) ##Mean pooling
        #x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class ResBlock_generator(nn.Module):
    def __init__(self, in_planes, planes, act_function=nn.ReLU(), norm=True, snorm=False, stride=1, config = None, z_dim=128):
        super(ResBlock_generator, self).__init__()
        self.act_function = act_function
        self.norm = norm
        self.snorm = snorm
        self.in_planes = in_planes
        self.planes = planes
        if self.snorm:
            self.conv1 = spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1))
            self.conv2 = spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        if self.norm:
            self.bn1 = adaptiveInstanceNorm2D(planes,s_norm_G=snorm,z_dim=z_dim)
            self.bn2 = adaptiveInstanceNorm2D(planes,s_norm_G=snorm,z_dim=z_dim)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()

        if in_planes != planes:
            if self.snorm:
                self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)))
            else:
                self.conv3 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0))
            if self.norm:
                self.bn3 = nn.adaptiveInstanceNorm2D(planes,s_norm_G=snorm,z_dim=z_dim)

    def forward(self, x, z):
        out = self.bn2(self.conv2( self.act_function(self.bn1(self.conv1(x),z))), z)
        if self.in_planes != self.planes:
            out = out + self.bn3(self.conv3(x), z)
        else:
            out = out + x
        out = self.act_function(out)
        return out

class adaptiveInstanceNorm2D(nn.Module):
    def __init__(self,planes,z_dim=128,s_norm_G=False):
        super(adaptiveInstanceNorm2D,self).__init__()
        self.bn = nn.BatchNorm2d(planes,affine = False)
        if s_norm_G:
            self.gamma = nn.Sequential(nn.Linear(z_dim,planes),nn.ReLU(),nn.Linear(planes,planes,bias=False))
            self.beta = nn.Sequential(nn.Linear(z_dim,planes),nn.ReLU(),nn.Linear(planes,planes,bias=False))
        else:
            self.gamma = nn.Sequential(spectral_norm(nn.Linear(z_dim,planes)),
            nn.ReLU(),spectral_norm(nn.Linear(planes,planes,bias=False)))
            self.beta = nn.Sequential(spectral_norm(nn.Linear(z_dim,planes)),
            nn.ReLU(),spectral_norm(nn.Linear(planes,planes,bias=False)))

    def forward(self, x, z):
        gamma_z = self.gamma(z).unsqueeze(2).unsqueeze(3)
        beta_z = self.beta(z).unsqueeze(2).unsqueeze(3)
        return gamma_z * self.bn(x) + beta_z

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, act_function=nn.LeakyReLU(0.2), norm=True, snorm=False, stride=1):
        super(ResBlock, self).__init__()
        self.act_function = act_function
        self.norm = norm
        self.snorm = snorm
        if self.norm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes), self.act_function)
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes))
        elif self.snorm:
            self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)),
                self.act_function)
            self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1),
                self.act_function)
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))

        if in_planes == planes:
            self.conv3 = nn.Sequential()
        else:
            if self.norm:
                self.conv3 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(planes))
            elif self.snorm:
                self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0)))
            else:
                self.conv3 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = out + self.conv3(x)
        out = self.act_function(out)
        return out

class CNN_cifar(nn.Module):
    def __init__(self, input_nc = 1, hidden_nc = 128):
        super(CNN_cifar, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_nc,
                out_channels=hidden_nc,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),
            nn.GELU(),
        )

        self.linear = nn.Sequential(nn.Linear(hidden_nc * 8 * 8, 100),
                                     nn.GELU())
        self.out = nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        output = self.out(x)
        return output, x    # return x for visualization

class CNN(nn.Module):
    def __init__(self, input_nc = 1, hidden_nc = 128):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_nc,
                out_channels=hidden_nc,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),
            nn.ReLU(),
        )
        if input_nc == 1:
            self.linear = nn.Sequential(nn.Linear(hidden_nc * 7 * 7, 100),
                                     nn.ReLU())
        else:
            self.linear = nn.Sequential(nn.Linear(hidden_nc * 8 * 8, 100),
                                     nn.ReLU())
        self.out = nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        output = self.out(x)
        return output, x    # return x for visualization

class Discriminator(nn.Module):
    def __init__(self, spectral_normalization, nclass=0):
        super(Discriminator, self).__init__()
        input_nc = 1
        hidden_nc = 512
        self.nclass = nclass

        if spectral_normalization:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride = 2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(hidden_nc, hidden_nc, kernel_size=4, stride = 2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride = 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=4, stride = 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        if self.nclass>0:
            self.linear = nn.Sequential(
                nn.Linear(hidden_nc*7*7+self.nclass, hidden_nc, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_nc, 1, bias=False),
            )
        else:
            if spectral_normalization:
                self.linear = nn.Sequential(spectral_norm(nn.Linear(hidden_nc*7*7,1,bias=False)))
            else:
                self.linear = nn.Sequential(nn.Linear(hidden_nc*7*7,1,bias=False))

    def forward(self, x, labels=None, device='cuda'):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        if self.nclass>0:
            labels = torch.nn.functional.one_hot(labels, self.nclass).cuda(device)
            x = torch.cat((x, labels), dim=1)
        x = self.linear(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class Mapping(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, depth = 1, linear = 0):
        super(Mapping, self).__init__()
        self.nc = nc
        layers = []
        for d in range(depth-1):
            layers.append(nn.Linear(z_dim,z_dim))
            if linear==0:
                layers.append(nn.LeakyReLU(0.2,inplace=True))
        layers.append(nn.Linear(z_dim, 7*7*nc))
        self.dense = nn.Sequential(*layers)

    def forward(self, z):
        z = self.dense(z)
        return z

class Synthesis(nn.Module):
    def __init__(self, nc = 128, tanh=1):
        super(Synthesis, self).__init__()
        self.nc = nc
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

        self.conv_final = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc, 1, kernel_size=3, padding=1))

        self.interpol_1 = Interpolate(size = (14,14), mode = 'nearest')
        self.interpol_2 = Interpolate(size = (28,28), mode = 'nearest')
        self.tanh = tanh
        self.final_activation = nn.Tanh()

    def forward(self, z):
        img = z.reshape(z.shape[0],self.nc,7,7)
        img = self.conv_1(img)
        img = self.interpol_1(img)
        img = self.conv_2(img)
        img = self.interpol_2(img)
        img = self.conv_final(img)
        if self.tanh==1:
            img = self.final_activation(img)
            img = img / 2 + 0.5
        return img

class Generator(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, depth = 1, tanh = 1, linear = 0, nclass=0):
        super(Generator, self).__init__()
        self.nc = nc
        self.nclass = nclass
        self.mapping = Mapping(z_dim+nclass, nc, depth, linear)
        self.synthesis = Synthesis(nc, tanh)

    def forward(self, z, labels=None, device='cuda'):
        if self.nclass>0:
            labels = torch.nn.functional.one_hot(labels, self.nclass).cuda(device)
            z = torch.cat((z, labels), dim=1)
        z = self.mapping(z)        
        img = self.synthesis(z)
        return img

def cal_gradient_penalty(netD, real_data, fake_data, device, labels=None, type='mixed', constant=1.0, 
                         lambda_gp=10.0, config = None):
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
            alpha = torch.rand(real_data.shape[0], 1).cuda(device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, labels.cuda(device))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def get_next_batch(dataiter, train_loader):
    try:
        images, labels = dataiter.next()
    except StopIteration:
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
    return images, labels, dataiter

def return_z(batch_size, z_dim):
    z = torch.randn((batch_size,z_dim))
    return z

def train(gen, optimizer_g, disc, optimizer_d, train_loader, grad_penalty, dataiter, device,name,d_step,z_dim):
    # Train the model
    for n in range(100001):
        for d in range(d_step):
            real, labels, dataiter = get_next_batch(dataiter, train_loader)
            real = real.cuda(device)
            fake = gen(return_z(real.shape[0],z_dim).cuda(device), labels)
            real_d, fake_d = disc(real, labels).mean(), disc(fake, labels).mean()
            emd = real_d - fake_d
            gradient_penalty, _ = cal_gradient_penalty(disc, real, fake, device, labels)
            loss = (- emd)
            if grad_penalty:
                loss+= gradient_penalty
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()

        for g in range(1):
            # gives batch data, normalize x when iterate train_loader
            z = return_z(real.shape[0],z_dim).cuda(device)
            fake = gen(z, labels)
            loss = - disc(fake, labels).mean()

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()

        if n%20 == 0:
            print('Step : ',str(n))
            print('real_d : ',real_d)
            print('fake_d : ',fake_d)
            print('gradient_penalty : ',gradient_penalty,flush=True)

        if n%500 == 0:
            figure = plt.figure(figsize=(5, 4))
            cols, rows = 10, 10
            for i in range(1, cols * rows + 1):
                img = (fake[i]).detach().cpu()
                figure.add_subplot(rows, cols, i)
                plt.axis("off")
                if img.shape[0] == 3:
                    img = img/2+0.5
                    img = rearrange(img,'c h w -> h w c')
                    plt.imshow(img.squeeze())
                else:
                    plt.imshow(img.squeeze(), cmap="gray")
            plt.savefig(name+'/gen_' + str(n) + '.png')

        if n%1000 == 0:
            path = name+'/gen_' + str(n) + '.pth'
            torch.save(gen.state_dict(), path)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=1,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--tanh",
        type=int,
        default=1,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--linear",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=0.00005,
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--d_step",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--gen",
        type=str,
        default="cnn",
    )
    return parser



def main():
    parser = get_parser()
    parser, unknown = parser.parse_known_args()
    print(parser)
    name = parser.name
    depth_g = parser.depth

    d_step = parser.d_step
    lr_g = parser.lr_g
    lr_d = parser.lr_d
    batch_size = 256
    z_dim = parser.z_dim
    grad_penalty = True
    spectral_normalization = False


    device = parser.device

    betas = (0.5,0.5)
    if parser.dataset == 'mnist':
        gen = Generator(z_dim=z_dim,nc=128,depth=depth_g,tanh=parser.tanh,linear=parser.linear).cuda(device)
        disc = Discriminator(spectral_normalization=spectral_normalization).cuda(device)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        train_data = datasets.MNIST(
            root = 'data',
            train = True,
            transform = ToTensor(),
            download = True,
        )
        test_data = datasets.MNIST(
            root = 'data',
            train = False,
            transform = ToTensor()
        )
    elif parser.dataset == 'synthetic_mnist_z2':
        gen = Generator(z_dim=z_dim,nc=128,depth=depth_g,tanh=parser.tanh,linear=parser.linear).cuda(device)
        disc = Discriminator(spectral_normalization=spectral_normalization).cuda(device)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        train_data = datasets.ImageFolder('data/' + parser.dataset + "/", \
                                          transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
        test_data = datasets.ImageFolder('data/' + parser.dataset + "/", \
                                         transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
        
    elif parser.dataset == 'mnist_conditional':
        gen = Generator(z_dim=z_dim,nc=128,depth=depth_g,tanh=parser.tanh,linear=parser.linear,nclass=10).cuda(device)
        disc = Discriminator(spectral_normalization=spectral_normalization, nclass=10).cuda(device)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        train_data = datasets.MNIST(
            root = 'data',
            train = True,
            transform = ToTensor(),
            download = True,
        )
        test_data = datasets.MNIST(
            root = 'data',
            train = False,
            transform = ToTensor()
        )
    elif parser.dataset == 'fmnist':
        gen = Generator(z_dim=z_dim,nc=128,depth=depth_g,tanh=parser.tanh,linear=parser.linear).cuda(device)
        disc = Discriminator(spectral_normalization=spectral_normalization).cuda(device)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        train_data = datasets.FashionMNIST(
            root = 'data',
            train = True,
            transform = ToTensor(),
            download = True,
        )
        test_data = datasets.FashionMNIST(
            root = 'data',
            train = False,
            transform = ToTensor()
        )
    elif parser.dataset == 'cifar10':
        batch_size=256
        betas = (0.,0.999)
        if parser.gen == 'style':
            gen = Generator_style(z_dim = z_dim, nc = 128, depth = depth_g, tanh = parser.tanh, linear = parser.linear).cuda(device)
            disc = Discriminator_style(spectral_normalization).cuda(device)
            optimizer_g = optim.Adam([{'params': gen.synthesis.parameters()},
                {'params': gen.mapping.parameters(), 'lr': lr_g*0.01}],
                                     lr = lr_g, betas = betas)
        else:
            gen = Generator_resnet(z_dim=z_dim,nc=128,tanh=parser.tanh).cuda(device)
            disc = Discriminator_resnet(snorm=spectral_normalization).cuda(device)
            optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    print(train_data)
    print(test_data)

    loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=15),

    'test'  : torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1),
    }

    train_loader = loaders['train']
    dataiter = iter(train_loader)
    print(gen)
    print(disc)
    train(gen, optimizer_g, disc, optimizer_d, train_loader, grad_penalty, dataiter, device, name, d_step,z_dim)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
