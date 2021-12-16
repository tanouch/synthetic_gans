# README #
Implementing synthetic WGANs. 
Base of the code for the following publications:
- Some Theoretical Insights into Wasserstein GANs (https://arxiv.org/abs/2006.02682)
- Approximating Lipschitz continuous functions with GroupSort neural networks (https://arxiv.org/abs/2006.05254)


## Main parameters
--loss_type: wgan-gp, hinge, vanilla, relativistic. Default: wgan-gp.
--gen_type: 'simple'.
--disc_type: 'simpleReLU' (ReLU networks), 'bjorckGroupSort' (as defined in https://arxiv.org/abs/1811.05381). Default: simpleReLU

### Others
--gen_lr, type = float, default=0.0025
--disc_lr, type = float, default=0.0025
--d_step', type=int, default=3
--g_step', type=int, default=1
--g_width', type=int, default=30
--d_width', type=int, default=30
--g_depth', type=int, default=2
--d_depth', type=int, default=5

Regarding the latent distribution:
--z_dim, default=1
--z_law, choices=('gauss', 'unif', 'circle'), default = 'unif'
--z_var', default=1.0

### Examples of some command runs: ###
python synthetic_gans/main.py --output_modes 4 --real_dataset_size 4 --z_dim 1 --name_exp 4modes_zdim1 --use_gpu 