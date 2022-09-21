from my_imports import *
from generating_data import create_mixture_gaussian_dataset
from plotting_functions import plot_gradient_of_the_generator, plot_densities, plot_heatmap_of_the_discriminator, \
    plot_heatmap_nearest_point, plot_densities_middle_points
from defining_models import Generator, Discriminator, Discriminator_bjorckGroupSort, Generator_mnist, Discriminator_mnist
from getting_pr_score import get_pr_scores, knn_scores
from tools import convert_to_gpu
from training_utils import train_discriminator, train_generator, save_models

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', type=str, default="default_exp")
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument("--device_id", type = int,  default = 0)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument("--use_gpu", action='store_true', help='shuffle input data')
    parser.add_argument("--spectral_normalization", action='store_true', help='shuffle input data')
    parser.add_argument("--spectral_normalization_iw", default=False, type=bool)
    parser.add_argument("--batch_norm_real_data", default=False, type=bool, help='batch norm input data')
    parser.add_argument('--steps_gan', type=int, default=20001)
    parser.add_argument('--steps_eval', type=int, default=1000)

    parser.add_argument("--loss_type", type = str, default = 'wgan-gp')
    parser.add_argument("--gen_type", type = str, default='simple')
    parser.add_argument("--disc_type", type = str, default='simple')
    parser.add_argument("--activation_function", type = str, default='relu')
    parser.add_argument("--gen_lr", type = float, default=0.0005)
    parser.add_argument("--disc_lr", type = float, default=0.001)
    parser.add_argument("--iw_lr", type = float, default=0.001)
    parser.add_argument("--iw_regul_weight", type = float, default=2.)
    parser.add_argument('--d_step', type=int, default=4)
    parser.add_argument('--g_step', type=int, default=1)
    parser.add_argument('--g_width', type=int, default=256)
    parser.add_argument('--d_width', type=int, default=256)
    parser.add_argument('--g_depth', type=int, default=2)
    parser.add_argument('--d_depth', type=int, default=5)
    parser.add_argument('--seed', type=int, default=50)

    #TRAINING
    #training_mode: do we define a training dataset (training) or we always sample from mu_star (test) ??
    parser.add_argument('--training_mode', type=str, default = "training")
    #testing_mode: we want to sample from mu_star so "test".
    parser.add_argument('--testing_mode', type=str, default = "test")
    #what metrics do we want to report ?
    parser.add_argument('--metrics', default = 'prec,rec,emd,hausd,emp_hausd', type=str)
    parser.add_argument('--plot_config', default = True, type=bool)
    parser.add_argument('--num_runs', default=1, type=int)
    parser.add_argument('--num_points_plotted', type=int, default=1000)

    #TRUE DIST
    #size of the training dataset: in few shot learning, we must have (real_dataset_size==output_modes)
    parser.add_argument('--real_dataset_size', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--output_modes_locs', default = 3, type=float)
    parser.add_argument('--output_modes', type=int, default=1)
    #variance of each mode of the Gaussian mixture
    parser.add_argument('--out_var', type=float, default=0.01)
    #dim of the latent space
    parser.add_argument('--z_dim', type=int, default=1)
    #law of the latent random variable
    parser.add_argument("--z_law", type = str, default = 'unif')
    #variance of the latent random variable
    parser.add_argument('--z_var', type=float, default=1.0)

    #KNN !!!!!
    #num_points to compute the the metrics: usually 2048...
    parser.add_argument('--num_points_sampled_knn', type=int, default=1024)
    #threshold to compute nearest_neighbors
    parser.add_argument('--kth_nearests', type=str, default='2')
    #num_runs to compute the confidence intervals for the metrics: usually 5...
    parser.add_argument('--num_runs_knn', default=5, type=int)
    parser.add_argument('--batch_size_knn', type=int, default=1024)
    parser.add_argument('--real_colors', type=str, default=None)

    opt = parser.parse_args()
    opt.kth_nearests = [int(item) for item in opt.kth_nearests.split(',')]
    if opt.real_colors is not None:
        opt.real_colors = [item for item in opt.real_colors.split(',')]
    opt.metrics = [item for item in opt.metrics.split(',')]
    opt.num_pics = 0
    return opt

config = get_config()
config.BCE = convert_to_gpu(nn.BCEWithLogitsLoss(), config)
config.results = dict()
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.output_modes != config.real_dataset_size:
    print("Careful, real_dataset_size different from output_modes")

if not os.path.exists(config.name_exp):
    os.makedirs(config.name_exp)

if config.dataset == 'synthetic':
    loc = config.output_modes_locs
    if config.output_dim==2:
        if config.output_modes == 1:
            config.means_mixture = [[0, 0]]
        elif config.output_modes == 3: #triangle equilateral !
            config.means_mixture = [[0, loc*1.732-1], [-loc, -1], [loc, -1]]
        elif config.output_modes == 4:
            config.means_mixture = [[-1*loc, 1*loc], [-1*loc, -1*loc], [1*loc, -1*loc], [1*loc, 1*loc]]
        elif config.output_modes == 5:
            config.means_mixture = [[-1*loc, 0.5*loc], [0, 1.*loc], [1*loc, 0.5*loc], [1*loc, -0.5*loc], [-1*loc,-0.5*loc]]
        elif config.output_modes == 6:
            config.means_mixture = [[-1, 1], [1, 1], [2, 0], [1, -1], [-1, -1], [-2, 0]]
        elif config.output_modes == 7:
            config.means_mixture = [[0*loc, -1*loc], [0, 0], [0*loc, 1*loc], \
                                    [1*loc, 1.5*loc], [2*loc, 2*loc], [-1*loc, 1.5*loc], [-2*loc,2*loc]]
        elif config.output_modes == 9:
            config.means_mixture = [[-1*loc, 1*loc],[0*loc, 1*loc],[1*loc, 1*loc],\
                                    [-1*loc, 0*loc],[0*loc, 0*loc],[1*loc, 0*loc], [-1*loc, -1*loc], [0*loc, -1*loc], [1*loc, -1*loc]]
        elif config.output_modes == 16:
            config.means_mixture = [[-1*loc, -2*loc], [-1*loc, -1*loc], [-1*loc, 0], [-1*loc, 1*loc], \
                                    [0, -2*loc], [0, -1*loc], [0, 0], [0, 1*loc], \
                                    [1*loc, -2*loc], [1*loc, -1*loc], [1*loc, 0], [1*loc, 1*loc], \
                                    [2*loc, -2*loc], [2*loc, -1*loc], [2*loc, 0], [2*loc, 1*loc]]
        else:
            config.means_mixture = [(np.random.rand(config.output_dim) - 0.5)*loc for i in range(config.output_modes)]
    weights = np.ones(config.output_modes)
    config.weights_mixture = weights/np.sum(weights)
    if config.training_mode=="training":
        config.real_dataset, config.real_dataset_index = create_mixture_gaussian_dataset(config), 0

elif config.dataset == "synthetic_simplex":
    config.output_dim = config.output_modes
    config.means_mixture = config.output_modes_locs*np.eye(config.output_modes)
    config.weights_mixture = np.ones(config.output_modes)/config.output_modes
    if config.training_mode=="training":
        config.real_dataset, config.real_dataset_index = create_mixture_gaussian_dataset(config), 0

else:
    print("Not the right dataset")
    sys.exit()

print(config, flush = True)
print("")
print("Starting training of GAN!")
if config.gen_type=="simple":
    generator = convert_to_gpu(Generator(config), config)
else:
    generator = convert_to_gpu(Generator_mnist(config), config)

if config.disc_type=="simple":
    discriminator = convert_to_gpu(Discriminator(config), config)
elif config.disc_type=="mnist":
    discriminator = convert_to_gpu(Discriminator_mnist(config), config)
else:
    discriminator = convert_to_gpu(Discriminator_bjorckGroupSort(config), config)

generator.train()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.gen_lr, betas=(0.5,0.5))
discriminator.train()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.disc_lr, betas=(0.5,0.5))

def get_scores_and_plot_graphs(metrics, generator, metric_score, mode, step, config):
    print("")
    print(metric_score, mode)
    dict_scores = get_pr_scores(metrics, config, generator, metric_score, mode)
    config.results[step] = dict_scores
    for metric in config.metrics:
        print(metric, dict_scores[metric])

for s in range(config.steps_gan):
    for d in range(config.d_step):
        train_discriminator(discriminator, generator, d_optimizer, s, config)
    for g in range(config.g_step):
        train_generator(discriminator, generator, g_optimizer, s, config)

    if s%config.steps_eval == 0 and s!= 0 :
        print('Steps', s)
        save_models(generator, s, "generator", config)
        save_models(discriminator, s, "discriminator", config)

        if config.dataset == 'synthetic' or config.dataset == 'synthetic_simplex':
            get_scores_and_plot_graphs(config.metrics, generator, metric_score="L2", mode=config.training_mode, step=s, config=config)
            z_means = knn_scores(generator, config)
            if (config.z_dim==2) or (config.z_dim==1):
                plot_heatmap_nearest_point(generator, config, span_length=config.z_var*1.5)
                plot_gradient_of_the_generator(generator, config, span_length=config.z_var*1.75, z_means=z_means)
            if (config.output_dim==2):
                plot_densities(config, generator)
                plot_densities_middle_points(config, generator)
            if (config.output_dim==2) and (config.z_dim==2):
                plot_heatmap_of_the_discriminator(discriminator, config)

        config.num_pics += 1
        print('____________________')

with open(config.name_exp+".json", 'w') as f:
    json.dump(config.results, f)
print("#######################")
