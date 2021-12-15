from my_imports import *
from tools import convert_to_gpu

def generate_z(batch_size, variance, config):
    mean = np.zeros(config.z_dim)
    cov = np.zeros((config.z_dim,config.z_dim))
    np.fill_diagonal(cov, variance)
    if config.z_law == "gauss":
        z = np.random.multivariate_normal(mean, cov, batch_size)
    elif config.z_law == "circle":
        z = np.random.multivariate_normal(mean, cov, batch_size)
        norm_z = np.linalg.norm(z, axis=1).reshape((-1, 1))
        z /= norm_z
    elif config.z_law == "segment":
        z = np.column_stack((np.random.uniform(low=-variance, high=variance, size=batch_size), np.zeros(batch_size)))
    elif config.z_law == "unif":
        z = np.random.uniform(low=-variance, high=variance, size=batch_size*config.z_dim).reshape((batch_size, config.z_dim))
    else:
        print("Not the right distribution for z")
        sys.exit()
    z = torch.from_numpy(z).float()
    return z

def generate_real_data(batch_size, config, mode):
    if mode=="training":
        if batch_size >= config.real_dataset_size:
            data = torch.cat([config.real_dataset]*int(batch_size/config.real_dataset_size) + \
                [config.real_dataset[:batch_size-config.real_dataset_size*int(batch_size/config.real_dataset_size)]])
        elif config.real_dataset_index + batch_size > config.real_dataset_size:
            data = config.real_dataset[config.real_dataset_index:]
            data = torch.cat([data, config.real_dataset[:config.real_dataset_size-config.real_dataset_index+batch_size]])
            config.real_dataset_index = config.real_dataset_size-config.real_dataset_index+batch_size
        else:
            data = config.real_dataset[config.real_dataset_index:config.real_dataset_index+batch_size]
            config.real_dataset_index += batch_size
        return data
    else:
        if config.dataset =='mnist' or config.dataset == 'fashionMNIST':
            return generate_mnist_batch(batch_size, config)
        else:
            return generate_synthetic_batch(batch_size, config, mode)

def get_next_batch(config):
    try:
        images, labels = config.dataiter.next()
    except StopIteration:
        config.dataiter = iter(config.train_loader)
        images, labels = config.dataiter.next()
    return images, labels

def generate_mnist_batch(num_points, config):
    if num_points == config.batch_size:
        real_images, real_labels = get_next_batch(config)
    else:
        batch_size = config.batch_size
        batch, labels = get_next_batch(config)
        real_images = torch.zeros((num_points,batch.shape[1],batch.shape[2],batch.shape[3]))
        real_labels = torch.zeros((num_points))
        real_images[:min(batch_size, num_points),:] = batch[:min(batch_size, num_points)]
        real_labels[:min(batch_size, num_points)] = labels[:min(batch_size, num_points)]

        index = min(batch_size, num_points)
        while index < num_points:
            batch, labels = get_next_batch(config)
            size = min(batch_size, num_points-index)
            batch, labels = batch[:size], labels[:size]
            real_images[index:index+size,:] = batch
            real_labels[index:index+size] = labels
            index += batch_size
    return real_images

def generate_synthetic_batch(batch_size, config, mode):
    cov = np.zeros((config.output_dim, config.output_dim))
    np.fill_diagonal(cov, config.out_var)
    samples = np.random.multinomial(batch_size, config.weights_mixture, size=1)[0]
    gauss = [np.random.multivariate_normal(config.means_mixture[i], cov, samples[i]) for i in range(config.output_modes)]
    gauss = [torch.from_numpy(gauss_i).float() for gauss_i in gauss]

    data = torch.FloatTensor()
    for gauss_i in gauss:
        data = torch.cat((data, gauss_i),0)

    if config.batch_norm_real_data:
        data[:,0] = (data[:,0] - torch.mean(data[:,0]))/torch.std(data[:,0])
        data[:,1] = (data[:,1] - torch.mean(data[:,1]))/torch.std(data[:,1])
    return data

def create_mixture_gaussian_dataset(config):
    if config.real_dataset_size==config.output_modes:
        data = torch.from_numpy(np.array(config.means_mixture)).float()
        return data
    else:
        cov = np.zeros((config.output_dim, config.output_dim))
        np.fill_diagonal(cov, config.out_var)
        samples = np.random.multinomial(config.real_dataset_size, config.weights_mixture, size=1)[0]
        gauss = [np.random.multivariate_normal(config.means_mixture[i], cov, samples[i]) for i in range(config.output_modes)]
        gauss = np.array(gauss)
        np.random.shuffle(gauss)
        gauss = [torch.from_numpy(gauss_i).float() for gauss_i in gauss]
        data = torch.FloatTensor()
        for gauss_i in gauss:
            data = torch.cat((data, gauss_i),0)
        data = data[torch.randperm(len(data))]
        return data


def rank_by_gradients(generator, num_points, config):
    num_gen_points = num_points
    num_points_batch = num_points
    nb_batches = 1
    nb_estimations = 10
    sigma = 10e-3

    norm_jacobian_set = convert_to_gpu(torch.FloatTensor(), config)
    gz_set = convert_to_gpu(torch.FloatTensor(), config)
    norm_jacobian = convert_to_gpu(torch.FloatTensor(num_points_batch), config)
    with torch.no_grad():
        for i in range(nb_batches):
            z = convert_to_gpu(generate_z(num_points_batch, config.z_var, config), config)
            if config.gen_type == 'dmlgan':
                gz, c_i = generator(z)
            else:
                gz = generator(z)
            gz_reshaped = gz.view(gz.shape[0], -1)
            multivar_normal = MultivariateNormal(torch.zeros(config.z_dim), sigma * torch.eye(config.z_dim))

            norm_jacobian.zero_()
            for n in range(nb_estimations):
                epsilon = multivar_normal.sample()
                epsilon = epsilon.view(1, epsilon.shape[0]).repeat(z.shape[0],1)
                epsilon = convert_to_gpu(epsilon, config)
                if config.gen_type == 'dmlgan':
                    gz_epsilon1 = generator.forward_fixed_gen(z + epsilon, c_i)
                    gz_epsilon2 = generator.forward_fixed_gen(z - epsilon, c_i)
                else:
                    gz_epsilon1 = generator(z + epsilon)
                    gz_epsilon2 = generator(z - epsilon)
                gz_epsilon1, gz_epsilon2 = gz_epsilon1.view(gz_epsilon1.shape[0], -1), gz_epsilon2.view(gz_epsilon2.shape[0], -1)
                norm_jacobian += torch.norm(gz_epsilon1 - gz_epsilon2, p=2, dim=1)/nb_estimations
            norm_jacobian_set = torch.cat((norm_jacobian_set, norm_jacobian))
            gz_set = torch.cat((gz_set, gz))
        gz_set = gz_set[torch.argsort(norm_jacobian_set)]
        return gz_set

def rank_by_discriminator(generator, discriminator, size, config):
    batch_size = config.batch_size_knn
    z = convert_to_gpu(generate_z(config.batch_size_knn, config.z_var, config), config)
    gz = generator(z)
    dz = discriminator(gz)
    n_batch = int(size/batch_size)
    for i in range(n_batch-1):
        z = convert_to_gpu(generate_z(config.batch_size_knn, config.z_var, config), config)
        gz_ = generator(z)
        gz = torch.cat((gz,gz_),0)
        dz_ = discriminator(gz_)
        dz = torch.cat((dz,dz_),0)
    fill_nb = size - (n_batch * batch_size)
    if fill_nb>0:
        z = convert_to_gpu(generate_z(fill_nb, config.z_var, config), config)
        gz_ = generator(z)
        gz = torch.cat((gz,gz_),0)
        dz_ = discriminator(gz_)
        dz = torch.cat((dz,dz_),0)
    dz = dz.squeeze(1)
    gz = gz[torch.argsort(-dz)]
    return gz