import os
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

from mnist_generation import CNN, CNN_cifar, Generator, Discriminator, get_next_batch, return_z, \
                                Generator_resnet, Generator_style

from torch import optim
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.stats
from scipy import linalg
from torchvision import datasets
from torchvision.transforms import ToTensor
from prdc import compute_prdc
from torchvision import transforms

#python eval.py --model_path cifar10_ganResnet_128th/gen_100000.pth --z_dim 128 --dataset cifar10 --tanh 1
#python eval.py --folder_path mnist_4 --gen_list gen_90000.pth gen_91000.pth gen_92000.pth gen_89000.pth  gen_88000.pth gen_87000.pth gen_86000.pth --z_dim 4 --dataset mnist
#python eval.py --folder_path mnist_6 --gen_list gen_90000.pth gen_91000.pth gen_92000.pth gen_89000.pth  gen_88000.pth gen_87000.pth gen_86000.pth --z_dim 6 --dataset mnist
#python eval.py --folder_path mnist_6 --gen_list gen_90000.pth gen_91000.pth gen_92000.pth gen_89000.pth  gen_88000.pth gen_87000.pth gen_86000.pth --z_dim 6 --dataset mnist
#python eval.py --folder_path mnist_8 --gen_list gen_90000.pth gen_91000.pth gen_85000.pth gen_89000.pth  gen_88000.pth gen_87000.pth gen_86000.pth --z_dim 8 --dataset mnist
#python eval.py --folder_path mnist_10 --gen_list gen_60000.pth gen_61000.pth gen_62000.pth gen_59000.pth  gen_58000.pth gen_57000.pth gen_56000.pth --z_dim 10 --dataset mnist
#python eval.py --folder_path mnist_12 --gen_list gen_90000.pth gen_91000.pth gen_85000.pth gen_89000.pth  gen_88000.pth gen_87000.pth gen_86000.pth --z_dim 12 --dataset mnist
#python eval.py --folder_path mnist_14 --gen_list gen_90000.pth gen_91000.pth gen_85000.pth gen_89000.pth  gen_88000.pth gen_87000.pth gen_86000.pth --z_dim 14 --dataset mnist
#python eval.py --folder_path mnist_synthetic_2 --gen_list gen_28000.pth gen_27000.pth gen_26000.pth gen_25000.pth  gen_24000.pth --z_dim 2 --dataset synthetic_mnist
#python eval.py --folder_path mnist_synthetic_8 --gen_list gen_28000.pth gen_27000.pth gen_26000.pth gen_25000.pth  gen_24000.pth --z_dim 8 --dataset synthetic_mnist
#python eval.py --folder_path mnist_synthetic_16 --gen_list gen_28000.pth gen_27000.pth gen_26000.pth gen_25000.pth  gen_24000.pth --z_dim 16 --dataset synthetic_mnist

@torch.no_grad()
def return_real_features(classifier,n_iterations,dataiter,train_loader):
    X = torch.tensor([])
    for i in range(n_iterations):
        img, _, _ = get_next_batch(dataiter, train_loader)
        img = img.cuda(0)
        with torch.no_grad():
            _,feats = classifier(img)
            #feats = img.view(img.shape[0],-1)
        X = torch.cat((X,feats.cpu()))
    return X
    
@torch.no_grad()
def return_gen_features(gen,classifier,batch_size,n_iterations,z_dim):
    X = torch.tensor([])
    for i in range(n_iterations):
        z = return_z(batch_size, z_dim).cuda(0)
        with torch.no_grad():
            gen_img = gen(z)
            _,feats = classifier(gen_img)
            #feats = gen_img.view(gen_img.shape[0],-1)
        X = torch.cat((X,feats.cpu()))
    return X

@torch.no_grad()
def return_filtered_gen_features(gen,classifier,batch_size,n_points,z_dim):
    X = torch.tensor([])
    while True:
        z = return_z(batch_size, z_dim).cuda(0)
        with torch.no_grad():
            gen_img = gen(z)
            scores,feats = classifier(gen_img)
            
            scores = torch.nn.functional.softmax(scores,dim=1)
            scores,_ = torch.max(scores,dim=1)
            zeros = torch.zeros_like(scores)
            idx_good_gen = torch.where(scores > 0.8, scores, zeros).nonzero()
            
            feats = feats[idx_good_gen].squeeze(1)
        X = torch.cat((X,feats.cpu()))
        if X.shape[0] > n_points:
            X = X[:n_points]
            break
    return X


def get_mu_sigma(feature_array):
    return np.mean(feature_array,axis=0), np.cov(feature_array,rowvar=False)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def print_format_metric(metric):
    metric_m, metric_s = mean_confidence_interval(metric)
    print('%.1f $\\pm$ %.1f ' % (metric_m, metric_s))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def return_x_y(gen,classifier,batch_size,n_iterations,z_dim):
    X, Y = [], []
    for i in range(n_iterations):
        z = return_z(batch_size, z_dim).cuda(0)
        with torch.no_grad():
            gen_img = gen(z)
            labels,_ = classifier(gen_img)
            labels = torch.max(labels, 1)[1].data.squeeze()
        X.append(z.detach().cpu().numpy())    
        Y.append(labels.detach().cpu().numpy())
    X,Y = np.concatenate(X,axis=0), np.concatenate(Y,axis=0)
    return X, Y

def return_x_y_filtered(gen,classifier,batch_size,n_points,z_dim):
    X, Y = [], []
    count = 0
    while True:
        z = return_z(batch_size, z_dim).cuda(0)
        with torch.no_grad():
            gen_img = gen(z)
            labels,_ = classifier(gen_img)
            
            max_score = torch.nn.functional.softmax(labels,dim=1)
            max_score,_ = torch.max(max_score,dim=1)
            zeros = torch.zeros_like(max_score)
            idx_good_gen = torch.where(max_score > 0.8, max_score, zeros).nonzero()
            
            z = z[idx_good_gen].squeeze(1)
            gen_img = gen_img[idx_good_gen].squeeze(1)
            labels = labels[idx_good_gen].squeeze(1)
            count += z.shape[0]
            
            labels = torch.max(labels, 1)[1].data.squeeze()
            
        X.append(z.detach().cpu().numpy())    
        Y.append(labels.detach().cpu().numpy())
        if count > n_points:
            X = X[:n_points]
            Y = Y[:n_points]
            break
    X,Y = np.concatenate(X,axis=0), np.concatenate(Y,axis=0)
    return X, Y

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
        "--folder_path",
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
    parser.add_argument('--gen_list', default=[], nargs='+')
    return parser

def main():
    parser = get_parser()
    parser, unknown = parser.parse_known_args()
    print(parser)
    dataset = parser.dataset
    z_dim = parser.z_dim
    gen = parser.gen
    device = parser.device
    folder_path = parser.folder_path
    gen_list = parser.gen_list
    gen_list = [os.path.join(folder_path,gen_path) for gen_path in gen_list]
    if dataset == 'cifar10':
        if gen == 'style':
            gen = Generator_style(z_dim = z_dim, nc = 128, depth = parser.depth, 
                                   tanh = parser.tanh, linear = parser.linear).cuda(parser.device)
            #gen.load_state_dict(torch.load(parser.model_path))
        else:
            tanh = 0
            gen = Generator_resnet(z_dim=z_dim,nc=128,tanh=tanh).cuda(0)
            #gen.load_state_dict(torch.load(parser.model_path))

        classifier = CNN_cifar(input_nc = 3, hidden_nc = 256).cuda(0)
        classifier.load_state_dict(torch.load('cifar10_classifier.pth'))
    else:
        gen = Generator(z_dim=z_dim,nc=128).cuda(0)
        #gen.load_state_dict(torch.load(parser.model_path))

        classifier = CNN().cuda(0)
        classifier.load_state_dict(torch.load('mnist_classifier.pth'))
    
    print('Logistic regression test')
    scores = []
    for path_gen in gen_list:
        print(path_gen)
        gen.load_state_dict(torch.load(path_gen))
        #print('construct dataset...')
        X, Y = return_x_y(gen,classifier,1000,100,z_dim)
        X_test, Y_test = return_x_y(gen,classifier,1000,10,z_dim)
        #print('training...')
        clf = LogisticRegression(penalty='none',solver='lbfgs',verbose=0).fit(X, Y)
        #print('testing...')
        sc = clf.score(X_test,Y_test)
        scores.append(sc*100)
        print(sc)
    print_format_metric(scores)
    
    print('Logistic regression on filtered generated points')
    scores = []
    for path_gen in gen_list:
        print(path_gen)
        gen.load_state_dict(torch.load(path_gen))
        print('construct dataset...')
        X, Y = return_x_y_filtered(gen,classifier,1000,1000*100,z_dim)
        X_test, Y_test = return_x_y_filtered(gen,classifier,1000,1000*10,z_dim)
        print('training...')
        clf = LogisticRegression(penalty='none',solver='lbfgs',verbose=0).fit(X, Y)
        print('testing...')
        sc = clf.score(X_test,Y_test)
        scores.append(sc*100)
        print(sc)
    print_format_metric(scores)
    
    print('Distribution fitting')
    batch_size = 250
    n_iter = 200
    if dataset == 'mnist':
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
    elif dataset == 'cifar10':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    elif dataset=='synthetic_mnist':
        train_data = datasets.ImageFolder(root='./data/synthetic_mnist_z2/', transform=transforms.Compose([transforms.Grayscale(),
                                                                                               transforms.ToTensor()]))
        test_data = datasets.ImageFolder(root='./data/synthetic_mnist_z2/', transform=transforms.Compose([transforms.Grayscale(),
                                                                                               transforms.ToTensor()]))


    train_loader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=batch_size, 
                                               shuffle=True, 
                                              num_workers=1) 
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=1)
    dataiter = iter(train_loader)

    prec, rec, dens, cov, fids = [],[],[],[],[]
    for path_gen in gen_list:
        print(path_gen)
        gen.load_state_dict(torch.load(path_gen))
        X_gen = return_gen_features(gen,classifier,batch_size,n_iter,z_dim).cpu().numpy()
        X_real = return_real_features(classifier,n_iter,dataiter,train_loader).cpu().numpy()
        print(X_gen.shape[0],X_real.shape[0])
        print('evaluating fid...')
        mu1, sigma1 = get_mu_sigma(X_real)
        mu2, sigma2 = get_mu_sigma(X_gen)
        fid = calculate_frechet_distance(mu1,sigma1,mu2,sigma2)
        print('evaluating prdc...')
        dict_prdc = compute_prdc(X_gen[:10000],X_real[:10000],nearest_k=5)
        prec.append(dict_prdc['precision']*100)
        rec.append(dict_prdc['recall']*100)
        dens.append(dict_prdc['density']*100)
        cov.append(dict_prdc['coverage']*100)
        fids.append(fid)
        print(dict_prdc,fid)

    print_format_metric(fids)
    print_format_metric(prec)
    print_format_metric(rec)
    print_format_metric(dens)
    print_format_metric(cov)
    
    print('Distribution fitting on filtered generations')
    
    prec, rec, dens, cov, fids = [],[],[],[],[]
    for path_gen in gen_list:
        print(path_gen)
        gen.load_state_dict(torch.load(path_gen))
        X_gen = return_filtered_gen_features(gen,classifier,batch_size,batch_size*n_iter,z_dim).cpu().numpy()
        X_real = return_real_features(classifier,n_iter,dataiter,train_loader).cpu().numpy()
        print(X_gen.shape[0],X_real.shape[0])
        print('evaluating fid...')
        mu1, sigma1 = get_mu_sigma(X_real)
        mu2, sigma2 = get_mu_sigma(X_gen)
        fid = calculate_frechet_distance(mu1,sigma1,mu2,sigma2)
        print('evaluating prdc...')
        dict_prdc = compute_prdc(X_gen[:10000],X_real[:10000],nearest_k=5)
        prec.append(dict_prdc['precision']*100)
        rec.append(dict_prdc['recall']*100)
        dens.append(dict_prdc['density']*100)
        cov.append(dict_prdc['coverage']*100)
        fids.append(fid)
        print(dict_prdc,fid)

    print_format_metric(fids)
    print_format_metric(prec)
    print_format_metric(rec)
    print_format_metric(dens)
    print_format_metric(cov)
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()