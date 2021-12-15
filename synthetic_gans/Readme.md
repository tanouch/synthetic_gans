# This is the code for the arxiv paper 
## This code trains WGANs architectures on few shot learning settings. 

Examples of some command runs:
python main.py --output_modes 4 --real_dataset_size 4 --z_dim 1 --name_exp 4modes_zdim1 --use_gpu 
python main.py --output_modes 4 --real_dataset_size 4 --z_dim 2 --name_exp 4modes_zdim2
python main.py --output_modes 9 --real_dataset_size 9 --z_dim 1 --name_exp 9modes_zdim1 
python main.py --output_modes 9 --real_dataset_size 9 --z_dim 2 --name_exp 9modes_zdim2
python main.py --output_modes 6 --real_dataset_size 6 --z_dim 1 --name_exp 6modes_zdim1 
python main.py --output_modes 6 --real_dataset_size 6 --z_dim 2 --name_exp 6modes_zdim2
python main.py --output_modes 15 --real_dataset_size 15 --z_dim 1 --name_exp 15modes_zdim1 
python main.py --output_modes 15 --real_dataset_size 15 --z_dim 2 --name_exp 15modes_zdim2