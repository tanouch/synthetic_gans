# README #
Implementing synthetic WGANs. 
Base of the code for the following publications:<br />
- Some Theoretical Insights into Wasserstein GANs (https://arxiv.org/abs/2006.02682).<br />
- Approximating Lipschitz continuous functions with GroupSort neural networks (https://arxiv.org/abs/2006.05254).<br />

### How to run it ??
1) Run install.sh
2) Activate venv with source venv/bin/activate

### Main parameters
--loss_type: wgan-gp, hinge, vanilla, relativistic. Default: wgan-gp.<br />
--gen_type: 'simple'.<br />
--disc_type: 'simpleReLU' (ReLU networks), 'bjorckGroupSort' (as defined in https://arxiv.org/abs/1811.05381). Default: simpleReLU.<br />
--use-gpu (significant speed-up) if you have access to one.

### Others
```--gen_lr, type = float, default=0.0025. <br />
--disc_lr, type = float, default=0.0025. <br />
--d_step, type=int, default=3.<br />
--g_step, type=int, default=1.<br />
--g_width, type=int, default=30.<br />
--d_width, type=int, default=30.<br />
--g_depth, type=int, default=2.<br />
--d_depth, type=int, default=5.<br />```

### Regarding the latent distribution:
```--z_dim, default=1<br />
--z_law, choices=('gauss', 'unif', 'circle'), default = 'unif'<br />
--z_var, default=1.0<br />```

## Examples of some command runs:
To train a WGANs on 4 data points, just run:
python synthetic_gans/main.py --dataset synthetic --output_modes 4 --real_dataset_size 4 --z_dim 1 --name_exp 4modes_zdim1 --use_gpu

## Examples of some command runs on MNIST & others:
```python mnist_generation.py --name mnist_2 --z_dim 2```
```python mnist_generation.py --name mnist_8 --z_dim 8```
```python mnist_generation.py --name mnist_2 --z_dim 2 --dataset mnist_conditional```
```python mnist_generation.py --name mnist_3 --z_dim 3 --dataset mnist_conditional```
```python mnist_generation.py --name mnist_synthetic_2 --z_dim 2 --dataset synthetic_mnist_z2```
```python mnist_generation.py --name mnist_synthetic_8 --z_dim 8 --dataset synthetic_mnist_z2```
```python mnist_generation.py --name mnist_synthetic_16 --z_dim 16 --dataset synthetic_mnist_z2```


## Implementations for real-life GANs (proposed by lucidrains on Github):
#https://github.com/lucidrains/lightweight-gan <br />
lightweight_gan --data ../data/celeba/celeba_2 --name celeba_2LG --aug-prob 0.25 --batch_size 16 --results_dir celeba_2LG/ --models_dir celeba_2LG/ --network-capacity 8 --num_train_steps 30001 --save_every 5000 --evaluate_every 5000 --image-size 256 <br />
lightweight_gan --data ../data/celeba/celeba_20 --name celeba_20LG --aug-prob 0.25 --batch_size 16 --results_dir celeba_20LG/ --models_dir celeba_20LG/ --network-capacity 8 --num_train_steps 20001 --save_every 5000 --evaluate_every 5000 --image-size 256


#https://github.com/lucidrains/stylegan2-pytorch <br />
stylegan2_pytorch --data ../data/celeba/celeba_500 --name celeba_500 --aug-prob 0.25 --multi-gpus --batch_size 16 --results_dir celeba_500/ --models_dir celeba_500/ --network-capacity 8 --num_train_steps 50000 --save_every 5000 --evaluate_every 5000 <br />
stylegan2_pytorch --data ../data/celeba/celeba_2 --name celeba_2 --aug-prob 0.25 --batch_size 16 --results_dir celeba_2/ --models_dir celeba_2/ --network-capacity 8 --num_train_steps 50000 --save_every 5000 --evaluate_every 5000
