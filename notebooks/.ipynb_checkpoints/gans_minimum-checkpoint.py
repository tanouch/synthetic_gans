import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
import sklearn.datasets
import time
from scipy.stats import truncnorm
from scipy.stats import norm
import matplotlib.pyplot as plt
import ot

opt_manualSeed = 15042017
print("Random Seed: ", opt_manualSeed)
random.seed(opt_manualSeed)
np.random.seed(opt_manualSeed)
torch.manual_seed(opt_manualSeed)
cudnn.benchmark = True
clip_fn = lambda x: x.clamp(max=0)

def inf_train_gen(DATASET, minn, maxx, nbdata, dim_out):
    loc, scale = 0.0, 1.
    if DATASET == 'uniform':
        dataset = np.random.rand(nbdata, dim_out)
        dataset = np.reshape(dataset, (nbdata, dim_out))
    
    if DATASET == 'trunc_norm':
        dataset = truncnorm.rvs((minn-loc)/scale, (maxx-loc)/scale, \
                                loc=loc, scale=scale, size=nbdata*dim_out)
        dataset = np.reshape(dataset, (nbdata, dim_out))
        
    if DATASET == 'norm':
        dataset = np.random.normal(loc, scale, size=nbdata*dim_out)
        dataset = np.reshape(dataset, (nbdata, dim_out))
        
    if DATASET == 'autre':
        centers = [1.1, 1.6, 2., 3.3, 4.3, 5.1, 6.5, 6.9, 7.2, 8.3]
        dataset = np.array(centers, dtype='float32')

    if DATASET == 'again':
        centers = [1., 2., 5., 5.5, 6.8, 9.0]
        dataset = np.array(centers, dtype='float32')
    return np.sort(dataset)

class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(MyLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    def forward(self, x):
        return torch.clamp(x, min=0.0)+torch.clamp(x, max=0.0)*self.negative_slope

class ToyGAN_G(nn.Module):
    def __init__(self, dim_hidden=64, dim_out=1, noise_dim=1):
        super(ToyGAN_G, self).__init__()
        self.dim_hidden, self.dim_out, self.noise_dim = dim_hidden, dim_out, noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_out)
            )
    def forward(self, x):
        x = self.net(x)
        return x

class ToyGAN_D(nn.Module):
    def __init__(self, dim_hidden=64, dim_out=1):
        super(ToyGAN_D, self).__init__()
        self.dim_hidden, self.dim_gen_out = dim_hidden, dim_out
        self.net = nn.Sequential(
            nn.Linear(dim_out, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            MyLeakyReLU(),
            nn.Linear(dim_hidden, 1)
            )
    def forward(self, x):
        x = self.net(x)
        return x.view(-1)


def cost_Matrix(x, y, dim_out):
    if dim_out==1:
        y = np.reshape(y, (-1,))
        M = np.zeros((len(x),len(y)))
        for i in range(len(x)):
            M[i] = np.abs(x[i]-y)
    else:
        M = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                M[i,j] = (x[i,0]-y[j,0])**2 + (x[i,1]-y[j,1])**2
    return M

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def create_generator_and_discriminator(nbdata, dim_out):
    netD = ToyGAN_D(dim_out=dim_out)
    netG = ToyGAN_G(dim_out=dim_out)
    netD.apply(init_weights)
    netG.apply(init_weights)

    netD.train()
    netG.train()
    x_to_w1 = torch.tensor(np.linspace(0,1,1000).reshape(1000,1))

    if opt_cuda:
        x_to_w1 = x_to_w1.cuda()
        netD.cuda()
        netG.cuda()
    return netG, netD, x_to_w1

def get_constraintK_and_minW(X):
    Wmin = 0
    Sup_diff=0
    for i in range(len(X)-1):
        Sup_diff = max(Sup_diff,X[i+1]-X[i])
        Wmin += (X[i+1]-X[i])**2/4
    K = 2*len(X)*Sup_diff + 10**-8
    Wmin /= K

    XXstar = np.zeros(len(X)+2)
    XXstar[0] = X[0]
    XXstar[1:-1] = X
    XXstar[-1] = X[-1]
    print('For K= ', K, 'we have Wmin=', Wmin)
    return K, Wmin, XXstar


def Gstar(u, XX, nbdata, K):
    k = int(nbdata*u)
    if u < k/nbdata + (XX[k+1]-XX[k])/2/K :
            psi = XX[k] + K*(u-(k/nbdata - (XX[k+1]-XX[k])/2/K))
    else :
        if u < (k+1)/nbdata - (XX[k+2]-XX[k+1])/2/K:
            psi = XX[k+1]
        else:
            psi = XX[k+1] + K*(u-((k+1)/nbdata - (XX[k+2]-XX[k+1])/2/K))
    return psi
    
def plot_both_generators_section3(step, X, XXstar, netG, data_to_plot, minn, maxx, LipG, dataset, K):
    def plot_both_generators_section3_aux(X, opposite_bool, suffix):
        X = np.sort(X)
        plt.clf()
        fig = plt.figure(figsize=(12,8))
        Z = np.array([int(1000*i/len(X))/1000 for i in range(1, len(X))])
        XX2 = np.array([(X[i+1]+X[i])/2 for i in range(len(X)-1)])
        plt.plot(Z, XX2,'o', label='Middles')

        Z = torch.tensor(np.linspace(0+10**-8, 1-10**-8, 1000).reshape(1000,1))
        if opt_cuda:
            Z= Z.cuda()
        if opposite_bool:
            Y = netG(1-Z.float())
        else:
            Y = netG(Z.float())
        Y = Y.cpu().detach().numpy().reshape(Y.shape[0])
        Z = Z.cpu().detach().numpy()
        psi_Z = np.zeros(1000) 
        for i in range(1000):
            psi_Z[i] = Gstar(Z[i], XXstar, len(X), K)

        plt.plot(Z, psi_Z, color='orange', label = 'Theoretical infinimum')
        plt.plot(Z, Y, color='blue', label = "Generative distribution")
        plt.yticks([])
        plt.legend(prop={'size':18}) 
        plt.savefig('img/learned_distrib_section3_'+ dataset + '_samples' + str(len(X)) \
                    + '_Lip' + str(int(LipG)) + '_step' + str(step) + suffix)
    plot_both_generators_section3_aux(X, opposite_bool=False, suffix='a')
    plot_both_generators_section3_aux(X, opposite_bool=True, suffix='b')
    
def plot_both_generators(step, X, XXstar, netG, data_to_plot, minn, maxx, LipG, dataset):
    X = np.sort(X)
    plt.clf()
    fig = plt.figure(figsize=(12,8))
    if dataset=='trunc_norm':
        minn, maxx = -1, 1
    else:
        minn, maxx = -2.5, 2.5
    bins = np.linspace(minn, maxx, 40)
    loc, scale = 0.0, 0.75
    if dataset=='trunc_norm':
        plt.plot(bins, truncnorm.pdf(bins, minn, maxx, loc, scale), 'r-', lw=3.5, label='True density')
    elif dataset=='norm':
        plt.plot(bins, norm.pdf(bins, loc, scale), 'r-', lw=3.5, label='True density')
    
    Z = torch.tensor(np.linspace(0+10**-8, 1-10**-8, 1000).reshape(1000,1))
    if opt_cuda:
        Z= Z.cuda()
    Y = netG(Z.float())
    Y = Y.cpu().detach().numpy().reshape(Y.shape[0])
    Z = Z.cpu().detach().numpy()
    plt.hist(Y, bins=bins, alpha=0.50, fill=True, density=True, color='blue', label="Generative distribution")
    bins_sticks = [bins[5*i] for i in range(int(len(bins)/5))]+[bins[-1]]
    plt.xticks(bins_sticks, fontsize=15)
    plt.yticks([])
    plt.legend(prop={'size':18}) 
    plt.savefig('img/learned_distrib_' + dataset + '_samples' + str(len(X)) \
                + '_Lip' + str(int(LipG)) + '_step' + str(step))

def plot_only_theoretical(X, XXstar, K, dataset):
    X = np.sort(X)
    plt.clf()
    fig = plt.figure(figsize=(12,8))
    Z = np.array([int(1000*i/len(X))/1000 for i in range(1, len(X))])
    XX2 = np.array([(X[i+1]+X[i])/2 for i in range(len(X)-1)])
    plt.plot(Z, XX2,'o', label='Middles')
    Z = torch.tensor(np.linspace(0+10**-8, 1-10**-8, 1000).reshape(1000,1))
    if opt_cuda:
        Z= Z.cuda()
    Z = Z.cpu().detach().numpy()
    psi_Z = np.zeros(1000) 
    for i in range(1000):
        psi_Z[i] = Gstar(Z[i], XXstar, len(X), K)
    
    plt.plot(Z, psi_Z, label = 'Theoretical infinimum')
    plt.yticks(np.array([X[i] for i in range(nbdata)]), np.array([X[i] for i in range(nbdata)]), \
              fontsize=15)
    plt.xticks(np.array([round(int(1000*i/nbdata)/1000,2) for i in range(nbdata+1)]), \
               np.array([round(int(1000*i/nbdata)/1000,2) for i in range(nbdata+1)]), fontsize=15)
    plt.legend(prop={'size':18}) 
    print('K = ',K)
    plt.savefig('img_theo/theoretical' + dataset + '_samples' + str(len(X)))
    
def train_GANs(Data, nbdata, XXstar, netG, netD, Lip_generator_c, \
               opt_lrD, opt_lrG, data_to_plot, minn, maxx, dataset, K, dim_out):
    optimizerD = optim.Adam(netD.parameters(), lr=opt_lrD)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_lrG)
    losses = []
    Batchs = opt_batchSize

    timet = time.time()
    for epochs in range(opt_niter):
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for j in range(opt_Diters):
            netD.zero_grad()
            data = Data[np.random.choice(len(Data), size=opt_batchSize, replace=True)]
            data = torch.tensor(data)
            U_data = data[torch.randperm(len(data))]
            interp_alpha = torch.rand(Batchs).view(Batchs,1)
            interp_points = interp_alpha*data +(1-interp_alpha)*U_data
            interp_points.requires_grad = True
            Z_data = torch.rand(len(data)).view(len(data), 1)
            
            if opt_cuda:
                data = data.cuda()
                U_data = U_data.cuda()
                interp_points = interp_points.cuda()
                Z_data = Z_data.cuda()
            
            errD_real = netD(data.float()).mean(0)
            fake = netG(Z_data)
            errD_fake = netD(fake.float()).mean(0)
            errD = errD_fake-errD_real

            errD_interp_vec = netD(interp_points.float())
            errD_gradient = torch.autograd.grad(errD_interp_vec.sum(), interp_points, create_graph=True)[0]
            lip_est_d = (errD_gradient**2).view(Batchs,-1).sum(1)**0.5
            lip_loss = opt_penalty_weight*(clip_fn(1.0-lip_est_d)**2).mean(0).view(1)
            errD = errD + lip_loss

            errD.backward()
            optimizerD.step()

        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        U_data = torch.rand(Batchs).view(Batchs,1)
        if opt_cuda:
            U_data = U_data.cuda()

        fake = netG(U_data)
        errG = netD(fake.float())
        errG = -errG.mean(0).view(1)

        if Lip_generator_c > 0:
            U_data.requires_grad = True
            errG_interp_vec = netG(U_data)
            errG_gradient = torch.autograd.grad(errG_interp_vec.sum(), U_data, create_graph=True)[0]
            lip_est_g = (errG_gradient**2).view(Batchs,-1).sum(1)**0.5
            lip_loss2 = 2000*opt_penalty_weight*(clip_fn(Lip_generator_c-lip_est_g)**2).mean(0).view(1)
            errG = errG + lip_loss2

        errG.backward()
        optimizerG.step()

        if (epochs) % opt_plot_every == 0:
            losses.append(errD.data[0])
            sizeGen= 2500
            x_to_w1 = torch.tensor(np.linspace(0, 1, sizeGen).reshape(sizeGen, 1)).cuda()
            XX = netG(x_to_w1.float())
            XX = XX.cpu().detach().numpy().reshape((XX.shape[0], dim_out))
            M = cost_Matrix(XX, Data, dim_out)
            a = (1/sizeGen)*np.ones(sizeGen)
            b = 1/nbdata*np.ones(len(Data))
            W1emp = ot.emd2(a, b, M)
            
            real_data = inf_train_gen(dataset, minn, maxx, sizeGen, dim_out)
            M = cost_Matrix(XX, real_data, dim_out)
            W1real = ot.emd2(a, a, M)
            

            print('Epoch [{}/{}], L_D: {:.4f}, L_G: {:.4f} , L_D_real: {:.4f} ,  L_D_fake:{:.4f}, LipD: {:.6f}, LipG: {:.10f}, W1emp: {:.4f}:, W1real: {:.4f}:  '.format(epochs, opt_niter,errD.data.item(), errG.data.item(), \
                errD_real.data.item(), errD_fake.data.item(), torch.max(lip_est_d).data.item(), \
                torch.max(lip_est_g).data.item(), W1emp, W1real))
            
            if dim_out==1:
                plot_both_generators(epochs, np.reshape(Data, (-1,)), XXstar, netG, data_to_plot, \
                                     minn, maxx, Lip_generator_c*10, dataset)
                plot_both_generators_section3(epochs, np.reshape(Data, (-1,)), XXstar, netG, data_to_plot, \
                                     minn, maxx, Lip_generator_c*10, dataset, K)
                 

dim_out = 2
opt_penalty_weight = 50.0 #penalty weight term lambda
opt_cuda = True
opt_clamp_lower = -0.01
opt_clamp_upper =  0.01
opt_Diters = 4
opt_niter = 12001
opt_batchSize = 64
opt_lrD = 0.0001
opt_lrG = 0.000075
opt_plot_every = 4000
minn, maxx = -1, 1
dataset_list = ['uniform']
nbdata_list = [5, 15, 50, 200, 1000]
Lip_generator_c_list = [0.5, 1, 5, 20, 50]
XXstar, K = None, None

for dataset in dataset_list:
    for nbdata in nbdata_list:
        Data = inf_train_gen(dataset, minn, maxx, nbdata, dim_out)
        data_to_plot = inf_train_gen(dataset, minn, maxx, 100000, dim_out)
        nbdata = len(Data)
        if dim_out==1:
            K, Wmin, XXstar = get_constraintK_and_minW(np.sort(np.reshape(Data, (-1,))))
            plot_only_theoretical(np.reshape(Data, (-1,)), XXstar, K, dataset)
        for Lip_generator_c in Lip_generator_c_list:
            print("###########################\n")
            print(dataset, 'ndata', str(nbdata), 'Lip', str(Lip_generator_c)) 
            netG, netD, x_to_w1 = create_generator_and_discriminator(nbdata, dim_out)
            train_GANs(Data, nbdata, XXstar, netG, netD, Lip_generator_c, \
                       opt_lrD, opt_lrG, data_to_plot, minn, maxx, dataset, K, dim_out)