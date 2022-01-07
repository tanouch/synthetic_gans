import os
import numpy as np
from shutil import copyfile


def create_dataset(folder_path, new_path, size):
    list_files = np.array(os.listdir(folder_path))
    selected_files = np.random.choice(list_files, size=size)
    print(selected_files)
    
    new_folder = new_path + 'celeba_' + str(size)
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    target_size = 2500
    for i in range(int(target_size/size)):
        for elem in selected_files:
            copyfile(folder_path + '/' + elem, new_folder+ '/' + str(i) + "_" + elem)

create_dataset('../data/celeba/celeba_cropped/', '../data/celeba/', size=2)
create_dataset('../data/celeba/celeba_cropped/', '../data/celeba/', size=4)
create_dataset('../data/celeba/celeba_cropped/', '../data/celeba/', size=20)
create_dataset('../data/celeba/celeba_cropped/', '../data/celeba/', size=100)
create_dataset('../data/celeba/celeba_cropped/', '../data/celeba/', size=500)