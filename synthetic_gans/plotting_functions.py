from my_imports import *
from matplotlib import colors
from tools import convert_to_gpu
from generating_data import generate_z, generate_real_data, rank_by_gradients, rank_by_discriminator

def generate_grid_z(xmin, xmax, ymin, ymax, num_points, dim):
    Z = []
    if dim==2:
        for x in np.linspace(xmin, xmax, num_points):
            for y in np.linspace(ymin, ymax, num_points):
                Z.append([x,y])
        return np.array(Z)
    if dim==1:
        for x in np.linspace(xmin, xmax, num_points):
            Z.append([x])
        return np.array(Z)

def calculate_norm(input_data, output, net, sigma, input_dim, nb_estimations, num_points, config, matrix):
    norm_gradients = convert_to_gpu(torch.zeros(input_data.size()[0]), config=config)
    multivar_normal = MultivariateNormal(torch.zeros(input_dim), sigma * torch.eye(input_dim))
    for n in range(nb_estimations):
        epsilon = multivar_normal.sample().view(1, input_dim).repeat(input_data.shape[0],1)
        epsilon = convert_to_gpu(epsilon, config)
        if config.gen_type == 'dmlgan':
            gz_epsilon1, c_i = net(input_data + epsilon)
            gz_epsilon2 = net.forward_fixed_gen(input_data - epsilon, c_i)
        else:
            gz_epsilon1 = net(input_data + epsilon)
            gz_epsilon2 = net(input_data - epsilon)
        gz_epsilon1, gz_epsilon2 = gz_epsilon1.view(gz_epsilon1.shape[0], -1), gz_epsilon2.view(gz_epsilon2.shape[0], -1)
        if matrix=="jacobian":
            norm_gradients += torch.norm(gz_epsilon1 - gz_epsilon2, p=2, dim=1)/nb_estimations
        elif matrix=="hessian":
            norm_gradients += torch.norm(gz_epsilon1 + gz_epsilon2 - 2*output, p=2, dim=1)/nb_estimations
        else:
            print("Not the right matrix argument")
            sys.exit()
    norm_gradients = norm_gradients.cpu().detach().numpy()
    min_ngrad, max_ngrad = min(norm_gradients), max(norm_gradients)
    if input_dim==2:
        norm_gradients = norm_gradients.reshape(num_points, num_points)
    if input_dim==1:
        norm_gradients = np.tile(norm_gradients, (len(norm_gradients), 1))
    return norm_gradients, min_ngrad, max_ngrad

def plot_gradient_of_the_generator(generator, config, span_length=2.5, num_points=100, z_means=None):
    if not os.path.exists(config.name_exp+'/gradients'):
        os.makedirs(config.name_exp+'/gradients')
    xmin, xmax, ymin, ymax = -span_length, span_length, -span_length, span_length
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points))
    z = generate_grid_z(xmin, xmax, ymin, ymax, num_points, config.z_dim)
    z = convert_to_gpu(torch.from_numpy(z).float(), config)
    if config.gen_type == 'dmlgan':
        gz, _ = generator(z)
    else:
        gz = generator(z)
    gz = gz.view(gz.shape[0], -1)

    norm, minn, maxx = calculate_norm(input_data=z, output=gz, net=generator, sigma=10e-3, \
                                      input_dim=config.z_dim, nb_estimations=10, num_points=num_points, config=config, matrix="jacobian")
    plt.clf()
    _, ax = plt.subplots()
    ax.pcolormesh(Xgrid, Ygrid, norm, vmin=minn, vmax=maxx, cmap='coolwarm', shading='auto')

    if z_means is not None:
        plt.scatter(0, 0, s=50,  alpha=1., c="k")
        plt.scatter(z_means[:,0], z_means[:,1], s=50,  alpha=1., c="g")

    ax.set_aspect('equal', 'datalim')
    plt.margins(0,0)
    ax.grid(False)
    plt.axis('off')
    plt.savefig(config.name_exp+"/gradients/norm_jacobian_"+str(config.num_pics)+".jpeg", bbox_inches="tight", pad_inches = 0)
    plt.savefig(config.name_exp+"/gradients/norm_jacobian_"+str(config.num_pics)+".pdf", bbox_inches="tight", pad_inches = 0)
    plt.close()

def calculate_distance_to_nearest_point(output, config, real_data=None):
    if real_data is None:
        if config.training_mode=="training":
            real_data = config.real_dataset
        else:
            real_data = config.means_mixture
    print("Here")
    print(output.shape, real_data.shape)
    distances = distance_matrix(output, real_data)
    print(distances.shape)
    matrix_distances = np.amin(distances, axis=1)
    classes = np.argmin(distances, axis=1)
    return matrix_distances, classes

def plot_heatmap_nearest_point(generator, config, span_length=2.5, num_points=250):
    if not os.path.exists(config.name_exp+'/distance_nearest_points'):
        os.makedirs(config.name_exp+'/distance_nearest_points')

    xmin, xmax, ymin, ymax = -span_length, span_length, -span_length, span_length
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points))
    z = generate_grid_z(xmin, xmax, ymin, ymax, num_points, dim=config.z_dim)
    z = convert_to_gpu(torch.from_numpy(z).float(), config)
    config.z_dim = z.shape[1]
    gz = generator(z)
    gz = gz.view(gz.shape[0], -1).detach().cpu().numpy()
    z = z.detach().cpu().numpy()

    norm, classes = calculate_distance_to_nearest_point(gz, config)
    z_Kmeans = list()
    for this_class in np.unique(classes):
        indexes = np.where(classes==this_class)[0]
        z_this_class = z[indexes]
        z_mean = np.mean(z_this_class, axis=0)
        z_mean /= np.linalg.norm(z_mean)
        z_Kmeans.append(z_mean)
    z_Kmeans = np.array(z_Kmeans)
    print("meanBIS", np.mean(z, axis=0))
    print('zmeansBIS', z_Kmeans)

    if config.z_dim==2:
        norm = norm.reshape(num_points, num_points)
        classes = classes.reshape(num_points, num_points)
    if config.z_dim==1:
        norm = np.tile(norm, (len(norm), 1))
        classes = np.tile(classes, (len(classes), 1))

    def plot_some_graph(norm, minn, maxx, classes, name, method):
        plt.clf()
        _, ax = plt.subplots()
        if method=="distance":
            ax.pcolormesh(Xgrid, Ygrid, norm, vmin=minn, vmax=maxx, cmap='coolwarm', shading='auto')
        else:
            if config.real_colors is not None:
                listed_cmap = colors.ListedColormap(config.real_colors)
            else:
                print("No colors defined per classes")
                return
            ax.imshow(classes, cmap=listed_cmap, origin='lower')
        ax.set_aspect('equal', 'datalim')
        plt.margins(0,0)
        ax.grid(False)
        plt.axis('off')
        plt.savefig(config.name_exp+"/distance_nearest_points/"+name+str(config.num_pics)+".jpeg", bbox_inches="tight", pad_inches = 0)
        plt.savefig(config.name_exp+"/distance_nearest_points/"+name+str(config.num_pics)+".png", bbox_inches="tight", pad_inches = 0)
        plt.savefig(config.name_exp+"/distance_nearest_points/"+name+str(config.num_pics)+".pdf", bbox_inches="tight", pad_inches = 0)
        plt.close()

    plot_some_graph(norm, np.amin(norm), np.amax(norm), classes, "distance_nearest_points_", method="distance")
    plot_some_graph(norm, np.amin(norm), np.amax(norm), classes, "class_nearest_points_", method="class")

def plot_heatmap_of_the_importance_weights(importance_weighter, config, span_length=2.5, num_points=100):
    if not os.path.exists(config.name_exp+'/importance_weights'):
        os.makedirs(config.name_exp+'/importance_weights')
    xmin, xmax, ymin, ymax = -span_length, span_length, -span_length, span_length
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points))
    z = generate_grid_z(xmin, xmax, ymin, ymax, num_points, config.z_dim)
    z = convert_to_gpu(torch.from_numpy(z).float(), config)
    wz = importance_weighter(z)
    wz = wz.detach().numpy()
    wz = np.maximum(wz, 0)
    min_wz, max_wz = min(wz), max(wz)
    wz = wz.reshape(num_points, num_points)
    _, ax = plt.subplots()
    ax.pcolormesh(Xgrid, Ygrid, wz, vmin=min_wz, vmax=max_wz)
    ax.set_aspect('equal', 'datalim')
    ax.grid(False)
    plt.axis('off')
    plt.savefig(config.name_exp+"/importance_weights/"+"importance_weights_"+str(config.num_pics)+".jpeg", bbox_inches="tight", pad_inches = 0)
    plt.savefig(config.name_exp+"/importance_weights/"+"importance_weights_"+str(config.num_pics)+".pdf", bbox_inches="tight", pad_inches = 0)
    plt.close()


def plot_gradient_of_the_discriminator(discriminator, config, num_points=100):
    if not os.path.exists(config.name_exp+'/disc_gradients'):
        os.makedirs(config.name_exp+'/disc_gradients')
    real_data = convert_to_gpu(generate_real_data(10000, config, mode="test"), config)
    real_data_np = real_data.detach().cpu().numpy()
    xmin, xmax, ymin, ymax = np.amin(real_data_np[:,0])-0.25, \
        np.amax(real_data_np[:,0])+0.25, np.amin(real_data_np[:,1])-0.25, np.amax(real_data_np[:,1])+0.25
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points))
    real_data_grid = generate_grid_z(xmin, xmax, ymin, ymax, num_points, config.z_dim)
    real_data_grid = convert_to_gpu(torch.from_numpy(real_data_grid).float(), config)
    dx = discriminator(real_data_grid)
    dx = dx.view(dx.shape[0], -1)
    norm, minn, maxx = calculate_norm(input_data=real_data_grid, \
                                      output=dx, net=discriminator, sigma=10e-3, input_dim=config.output_dim, \
                                      nb_estimations=10, num_points=num_points, config=config, matrix="jacobian")
    plt.clf()
    _, ax = plt.subplots()
    ax.pcolormesh(Xgrid, Ygrid, norm, vmin=minn, vmax=maxx, cmap='coolwarm')
    ax.set_aspect('equal', 'datalim')
    plt.margins(0,0)
    ax.grid(False)
    plt.axis('off')
    plt.savefig(config.name_exp+"/disc_gradients/norm_jacobian_"+str(config.num_pics)+".jpeg", bbox_inches="tight", pad_inches = 0)
    plt.savefig(config.name_exp+"/disc_gradients/norm_jacobian_"+str(config.num_pics)+".pdf", bbox_inches="tight", pad_inches = 0)
    plt.close()


def plot_heatmap_of_the_discriminator(discriminator, config, num_points=100):
    if not os.path.exists(config.name_exp+'/disc_heatmaps'):
        os.makedirs(config.name_exp+'/disc_heatmaps')
    real_data = convert_to_gpu(generate_real_data(10000, config, mode="test"), config)
    real_data_np = real_data.detach().cpu().numpy()
    xmin, xmax, ymin, ymax = np.amin(real_data_np[:,0])-0.25, np.amax(real_data_np[:,0])+0.25, np.amin(real_data_np[:,1])-0.25, np.amax(real_data_np[:,1])+0.25
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points))
    real_data_grid = generate_grid_z(xmin, xmax, ymin, ymax, num_points, config.z_dim)
    real_data_grid = convert_to_gpu(torch.from_numpy(real_data_grid).float(), config)
    dx = discriminator(real_data_grid)
    dx = dx.view(dx.shape[0], -1)
    dx = dx.cpu().detach().numpy()
    minn, maxx = min(dx), max(dx)
    dx = dx.reshape(num_points, num_points)
    plt.clf()
    _, ax = plt.subplots()
    ax.pcolormesh(Xgrid, Ygrid, dx, vmin=minn, vmax=maxx, cmap='coolwarm', shading='auto')
    ax.set_aspect('equal', 'datalim')
    plt.margins(0,0)
    ax.grid(False)
    plt.axis('off')
    plt.savefig(config.name_exp+"/disc_heatmaps/"+str(config.num_pics)+".jpeg", bbox_inches="tight", pad_inches = 0)
    plt.savefig(config.name_exp+"/disc_heatmaps/"+str(config.num_pics)+".pdf", bbox_inches="tight", pad_inches = 0)
    plt.close()


def plot_data_points(generator, config):
    z = convert_to_gpu(generate_z(config.num_points_plotted, config.z_var, config), config)
    gz = generator(z)
    gz_np = gz.detach().cpu().numpy()
    real_data = convert_to_gpu(generate_real_data(len(gz), config, mode="test"), config)
    real_data_np = real_data.detach().cpu().numpy()
    plt.clf()
    plt.figure(frameon=False)
    plt.plot(gz_np[:,0], gz_np[:,1], 'b.', alpha=0.90, label="Fake")
    plt.plot(real_data_np[:,0], real_data_np[:,1], "r+", alpha=0.35, label="Real")
    if not os.path.exists(config.name_exp+'/data_points'):
        os.makedirs(config.name_exp+'/data_points')
    plt.axis('off')
    plt.savefig(config.name_exp+'/data_points'+'/data_points_'+str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.savefig(config.name_exp+'/data_points'+'/data_points_'+str(config.num_pics)+".pdf", bbox_inches="tight")
    plt.close()


def plot_densities(config, generator=None):
    if config.training_mode=="training":
        emp_np = np.array(config.real_dataset)
    else:
        emp_np = np.array(config.means_mixture)
        #emp_np = generate_real_data(config.num_points_plotted, config, config.training_mode).detach().cpu().numpy()

    gz_np = generator(convert_to_gpu(generate_z(config.num_points_plotted, config.z_var, config), config)).detach().cpu().numpy()
    xmin, xmax, ymin, ymax = \
        min(np.amin(emp_np[:,0])-0.5, np.amin(gz_np[:,0])-0.5), \
        max(np.amax(emp_np[:,0])+0.5, np.amax(gz_np[:,0])+0.5), \
        min(np.amin(emp_np[:,1])-0.5, np.amin(gz_np[:,1])-0.5), \
        max(np.amax(emp_np[:,1])+0.5, np.amax(gz_np[:,1])+0.5)

    plt.clf()
    _, ax = plt.subplots()
    plt.figure(frameon=False)
    plt.scatter(gz_np[:,0], gz_np[:,1], s=3,  alpha=0.25, c="c")
    if (config.real_colors is not None) and (config.training_mode=="training"):
        for i in range(len(emp_np)):
            plt.scatter(emp_np[i][0], emp_np[i][1], s=75, alpha=0.85, c=config.real_colors[i])
    else:
          plt.scatter(emp_np[:,0][:512], emp_np[:,1][:512], s=50, alpha=0.85, c="r")
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    ax.set_aspect('equal', 'datalim')
    plt.axis('off')
    if not os.path.exists(config.name_exp+'/densities'):
        os.makedirs(config.name_exp+'/densities')
    plt.savefig(config.name_exp+'/densities'+'/densities_'+str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.savefig(config.name_exp+'/densities'+'/densities_'+str(config.num_pics)+".pdf", bbox_inches="tight")
    plt.close()

def plot_densities_middle_points(config, generator=None):
    emp_np = generate_real_data(config.num_points_plotted, config, config.training_mode).detach().cpu().numpy()
    gz_np_mid = generator(convert_to_gpu(generate_z(int(config.num_points_plotted/5), \
                                                              config.z_var/50, config), config)).detach().cpu().numpy()
    gz_np = generator(convert_to_gpu(generate_z(config.num_points_plotted, config.z_var, config), config)).detach().cpu().numpy()
    xmin, xmax, ymin, ymax = \
        min(np.amin(emp_np[:,0])-0.5, np.amin(gz_np[:,0])-0.5), \
        max(np.amax(emp_np[:,0])+0.5, np.amax(gz_np[:,0])+0.5), \
        min(np.amin(emp_np[:,1])-0.5, np.amin(gz_np[:,1])-0.5), \
        max(np.amax(emp_np[:,1])+0.5, np.amax(gz_np[:,1])+0.5)

    plt.clf()
    fig, ax = plt.subplots()
    plt.figure(frameon=False)
    plt.scatter(gz_np[:,0], gz_np[:,1], s=15,  alpha=0.5, c="k")
    plt.scatter(gz_np_mid[:,0], gz_np_mid[:,1], s=15,  alpha=0.5, c="g")
    plt.scatter(emp_np[:,0][:512], emp_np[:,1][:512], s=15,  alpha=0.95, c="r")
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    ax.set_aspect('equal', 'datalim')
    plt.axis('off')
    if not os.path.exists(config.name_exp+'/densities'):
        os.makedirs(config.name_exp+'/densities')
    plt.savefig(config.name_exp+'/densities'+'/densities_midpoints_'+str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.savefig(config.name_exp+'/densities'+'/densities_midpoints_'+str(config.num_pics)+".pdf", bbox_inches="tight")
    plt.close()


def plot_densities_iw(config, generator, importance_weighter):
    z = generate_z(config.num_points_plotted, config.z_var, config)
    wz = importance_weighter(z)
    gz = generator(z).detach()
    gz_np = gz.cpu().numpy()
    wz_np = wz.detach().cpu().numpy()
    wz_np = np.maximum(wz_np, 0)
    x, y = gz_np[:,0], gz_np[:,1]
    plt.clf()
    real_data = generate_real_data(config.num_points_plotted, config, config.training_mode)
    real_data_np = real_data.detach().numpy()
    plt.scatter(real_data_np[:,0], real_data_np[:,1], s=15, alpha= 0.5, c='r')
    for p in range(config.num_points_plotted):
        color = [0, 0, 1, min(1, wz_np[p, 0])]
        plt.scatter(gz_np[p,0],gz_np[p,1], s=15, color=color)
    plt.axis('off')
    if not os.path.exists(config.name_exp+'/densities'):
        os.makedirs(config.name_exp+'/densities')
    plt.savefig(config.name_exp+'/densities'+'/densities_importance_weighter_'+str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.savefig(config.name_exp+'/densities'+'/densities_importance_weighter_'+str(config.num_pics)+".pdf", bbox_inches="tight")
    plt.close()


def plot_mnist_images(generator, config):
    nb_lines, nb_cols = 10, 10
    z = convert_to_gpu(generate_z(1000, config.z_var, config), config)
    gz = generator(z)
    he, wi = 28, 28
    gz = gz.view(gz.shape[0], -1)
    gz_np = gz.detach().cpu().numpy()
    gz_np = np.reshape(gz_np,(gz_np.shape[0],he,wi))
    image = np.zeros((he*nb_lines,wi*nb_cols))
    for i, gz in enumerate(gz_np):
        if i == nb_lines*nb_cols:
            break
        h, w = i%nb_lines, i//nb_cols
        image[h*he:h*he+he,w*wi:w*wi+wi] = - gz + 1
    plt.clf()
    plt.figure(figsize=(20,20))
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    plt.savefig(config.name_exp+'/img_'+ str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.close()


def plot_mnist_ranked_images(generator, config):
    if not os.path.exists(config.name_exp+'/ranked_images'):
        os.makedirs(config.name_exp+'/ranked_images')
    nb_lines, nb_cols = 10, 10
    gz = rank_by_gradients(generator, nb_lines*nb_cols, config)
    he, wi = 28, 28
    gz = gz.view(gz.shape[0], -1)
    gz_np = gz.detach().cpu().numpy()
    gz_np = np.reshape(gz_np,(gz_np.shape[0],he,wi))

    image = np.zeros((he*nb_lines,wi*nb_cols))
    for i,gz in enumerate(gz_np):
        if i == nb_lines*nb_cols:
            break
        h, w = i//nb_lines, i%nb_cols
        image[h*he:h*he+he,w*wi:w*wi+wi] = - gz + 1

    plt.clf()
    plt.figure(figsize=(20,20))
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    plt.savefig(config.name_exp+'/ranked_images/ranked_images_'+str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.close()


def plot_mnist_last_ranked_images(generator, discriminator, ranking, config):
    if not os.path.exists(config.name_exp+'/ranked_images'):
        os.makedirs(config.name_exp+'/ranked_images')
    nb_lines, nb_cols = 10, 10
    if ranking=="gradient":
        gz = rank_by_gradients(generator, 1000, config)[-100:]
    elif ranking=="discriminator":
        gz = rank_by_discriminator(generator, discriminator, 1000, config)[-100:]
    else:
        print("Not the right ranking")
        sys.exit()

    he, wi = 28, 28
    gz = gz.view(gz.shape[0], -1)
    gz_np = gz.detach().cpu().numpy()
    gz_np = np.reshape(gz_np,(gz_np.shape[0],he,wi))
    image = np.zeros((he*nb_lines,wi*nb_cols))
    for i,gz in enumerate(gz_np):
        if i == nb_lines*nb_cols:
            break
        h, w = i//nb_lines, i%nb_cols
        image[h*he:h*he+he,w*wi:w*wi+wi] = - gz + 1

    plt.clf()
    plt.figure(figsize=(20,20))
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    plt.savefig(config.name_exp+'/ranked_images/last_ranked_images_'+ranking+str(config.num_pics)+".jpeg", bbox_inches="tight")
    plt.close()


def plot_in_between_modes_MST(generator, discriminator, ranking, metric, config):
    if not os.path.exists(config.name_exp+'/position_data'):
        os.makedirs(config.name_exp+'/position_data')

    if ranking=="gradient":
        gz = rank_by_gradients(generator, 750, config)[-35:]
    elif ranking=="discriminator":
        gz = rank_by_discriminator(generator, discriminator, 750, config)[-35:]
    else:
        print("Not the right ranking")
        sys.exit()

    gz = gz.view(gz.shape[0], 1, int(np.sqrt(gz.shape[1])), int(np.sqrt(gz.shape[1])))
    data = torch.cat((config.real_dataset, gz))

    if metric=="classifier":
        data, _ = config.classifier(data)
    elif metric=="L2":
        data = data.reshape((data.shape[0], -1))
    else:
        print("Not the right metric")
        sys.exit()

    data = data.detach().numpy()
    T = find_MST_networkx(data)
    plt.clf()
    plt.axis('off')
    pos = nx.spring_layout(T)
    pos_labels, labels = {}, {}
    for node in list(T.nodes())[:config.real_dataset_size]:
        labels[node] = config.real_dataset_labels[node]
        pos_labels[node] = pos[node]

    nx.draw_networkx_nodes(T, pos, nodelist= list(T.nodes())[config.real_dataset_size:] , node_color='b', node_size=125, alpha=0.60)
    nx.draw_networkx_nodes(T, pos, nodelist = list(T.nodes())[:config.real_dataset_size], node_color='r', node_size=500, alpha=1.0)
    nx.draw_networkx_labels(T, pos_labels, labels, font_size=20, font_color='k')
    nx.draw_networkx_edges(T, pos)
    plt.savefig(config.name_exp+"/position_data/MST_"+str(config.num_pics)+"_"+metric+"_"+ranking, bbox_inches="tight")
    plt.savefig(config.name_exp+"/position_data/MST_"+str(config.num_pics)+"_"+metric+"_"+ranking+".pdf", bbox_inches="tight")


def plot_mnist_dataset(config):
    gz = config.real_dataset
    nb_lines, nb_cols = 1, int(gz.shape[0])
    he, wi = 28, 28
    gz = gz.view(gz.shape[0], -1)
    gz_np = gz.detach().cpu().numpy()
    gz_np = np.reshape(gz_np,(gz_np.shape[0], he, wi))
    image = np.zeros((he*nb_lines,wi*nb_cols))
    for i, gz in enumerate(gz_np):
        if i == nb_lines*nb_cols:
            break
        image[0:he,i*wi:(i+1)*wi] = - gz + 1
    plt.clf()
    plt.figure(figsize=(5*nb_cols, 5))
    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    plt.savefig(config.name_exp+"/real_images.jpeg")
    plt.close()


def plot_in_between_modes(generator, metric, config):
    if not os.path.exists(config.name_exp+'/position_data'):
        os.makedirs(config.name_exp+'/position_data')

    gz = rank_by_gradients(generator, 750, config)[-35:]
    gz = gz.view(gz.shape[0], 1, int(np.sqrt(gz.shape[1])), int(np.sqrt(gz.shape[1])))
    data = torch.cat((config.real_dataset, gz))

    if metric=="classifier":
        data, _ = config.classifier(data)
    elif metric=="L2":
        data = data.reshape((data.shape[0], -1))
    else:
        print("Not the right metric")
        sys.exit()

    data = data.detach().numpy()
    data = cdist(data, data, metric="euclidean")
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=123, n_init=10, n_jobs=-1, max_iter=500)
    results = mds.fit(data)
    coords = results.embedding_

    plt.clf()
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(coords[:, 0][:config.real_dataset_size], coords[:, 1][:config.real_dataset_size], marker = 'o', c="b")
    plt.scatter(coords[:, 0][config.real_dataset_size:], coords[:, 1][config.real_dataset_size:], marker = 'o', c="r")
    plt.axis('off')

    matches = np.array(find_MST(coords[:config.real_dataset_size]))
    for i in range(len(matches)):
        x, y = [coords[matches[:,0][i]][0], coords[matches[:,1][i]][0]], [coords[matches[:,0][i]][1], coords[matches[:,1][i]][1]]
        plt.plot(x, y, c="b")

    for label, x, y in zip(config.real_dataset_labels, coords[:, 0][:config.real_dataset_size], coords[:, 1][:config.real_dataset_size]):
        plt.annotate(str(label)[7:-1], xy = (x, y), xytext = (-20, 20), fontsize=15,
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.savefig(config.name_exp+"/position_data/position_data_"+str(config.num_pics)+"_"+metric, bbox_inches="tight")
    plt.savefig(config.name_exp+"/position_data/position_data_"+str(config.num_pics)+"_"+metric+".pdf", bbox_inches="tight")
