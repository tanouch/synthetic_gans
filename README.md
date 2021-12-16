# README #
Implementing synthetic WGANs. 
Base of the code for the following publications:<br />
- Some Theoretical Insights into Wasserstein GANs (https://arxiv.org/abs/2006.02682).<br />
- Approximating Lipschitz continuous functions with GroupSort neural networks (https://arxiv.org/abs/2006.05254).<br />


## Main parameters
--loss_type: wgan-gp, hinge, vanilla, relativistic. Default: wgan-gp.<br />
--gen_type: 'simple'.<br />
--disc_type: 'simpleReLU' (ReLU networks), 'bjorckGroupSort' (as defined in https://arxiv.org/abs/1811.05381). Default: simpleReLU.<br />

### Others
--gen_lr, type = float, default=0.0025. <br />
--disc_lr, type = float, default=0.0025. <br />
--d_step', type=int, default=3.<br />
--g_step', type=int, default=1.<br />
--g_width', type=int, default=30.<br />
--d_width', type=int, default=30.<br />
--g_depth', type=int, default=2.<br />
--d_depth', type=int, default=5.<br />

Regarding the latent distribution:
--z_dim, default=1<br />
--z_law, choices=('gauss', 'unif', 'circle'), default = 'unif'<br />
--z_var', default=1.0<br />

### Examples of some command runs: ###
python synthetic_gans/main.py --output_modes 4 --real_dataset_size 4 --z_dim 1 --name_exp 4modes_zdim1 --use_gpu 