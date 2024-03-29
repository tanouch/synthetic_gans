from my_imports import *
from tools import convert_to_gpu
from generating_data import generate_z, generate_real_data
from plotting_functions import calculate_distance_to_nearest_point

def get_scores_and_plot_graphs(metrics, generator, metric_score, mode, step, config):
    print("")
    print(metric_score, mode)
    dict_scores = get_pr_scores(metrics, config, generator, metric_score, mode)
    config.results[step] = dict_scores
    for metric in config.metrics:
        print(metric, dict_scores[metric])

def get_conf_int(data):
    if (np.amax(data)==np.amin(data)):
        return (data[0], data[0])
    else:
        return norm.interval(0.95, loc=np.mean(np.array(data)), scale=np.std(np.array(data))/math.sqrt(len(data)))

def get_pr_scores(metrics, config, generator=None, metric_score="L2", mode="test"):
    dict_metrics, precisions, recalls, emds, hausds, emp_hausds = dict(), list(), list(), list(), list(), list()
    for i in range(config.num_runs_knn):
        #GENERATOR
        if generator==None:
            set_gz = generate_real_data(config.real_dataset_size, config, config.training_mode)
        else:
            set_gz = generator(convert_to_gpu(generate_z(config.num_points_sampled_knn, config.z_var, config), config))

        #MODE
        if mode=="test":
            set_real_data = convert_to_gpu(generate_real_data(config.num_points_sampled_knn, config, mode), config)
        else:
            set_real_data = convert_to_gpu(generate_real_data(config.real_dataset_size, config, mode), config)

        #METRIC_SCORE
        if metric_score=="classifier":
            if config.dataset == 'mnist':
                with torch.no_grad():
                    if config.gen_type=='simple':
                        set_gz = set_gz.view(config.num_points_sampled_knn, 1, int(math.sqrt(config.output_dim)), int(math.sqrt(config.output_dim)))
                    set_gz, _ = config.classifier(set_gz)
                    set_real_data, _ = config.classifier(set_real_data)

        set_gz = set_gz.view(set_gz.shape[0], -1)
        set_real_data = set_real_data.view(set_real_data.shape[0], -1)

        for metric in metrics:
            if metric=="prec":  precisions.append(manifold_estimate(set_real_data, set_gz, config))
            if metric=="rec":   recalls.append(manifold_estimate(set_gz, set_real_data, config))
            if metric=="hausd": hausds.append(max(hausdorff_estimate(set_gz, set_real_data, config), hausdorff_estimate(set_real_data, set_gz, config)))
            if metric=="emp_hausd": emp_hausds.append(hausdorff_estimate(set_real_data, set_gz, config))
            if metric=="emd":
                if config.use_gpu == True:
                    s1, s2 = set_gz.detach().cpu().numpy(), set_real_data.detach().cpu().numpy()
                else:
                    s1, s2 = set_gz.detach().numpy(), set_real_data.detach().numpy()
                a, b = np.ones((len(s1),))/len(s1), np.ones((len(s2),))/len(s2)
                M = ot.dist(s1, s2, metric='euclidean')
                max_m = M.max()
                M /= max_m
                emd = ot.emd2(a, b, M, numItermax=100000)
                emds.append(emd*max_m)

    for metric in metrics:
        if metric=="knn":  dict_metrics[metric] = knn_scores(generator, config)
        if metric=="prec":  dict_metrics[metric] = get_conf_int([elem[0] for elem in precisions])
        if metric=="rec":   dict_metrics[metric] = get_conf_int([elem[0] for elem in recalls])
        if metric=="emd":   dict_metrics[metric] = get_conf_int(emds)
        if metric=="hausd":   dict_metrics[metric] = get_conf_int(hausds)
        if metric=="emp_hausd":   dict_metrics[metric] = get_conf_int(emp_hausds)
    return dict_metrics


def manifold_estimate(X_a, X_b, config):
    size_seta, size_setb, batch_size = X_a.shape[0], X_b.shape[0], config.batch_size_knn

    k_th_distance = convert_to_gpu(torch.zeros((size_seta, len(config.kth_nearests))), config)
    pairwise_distances = convert_to_gpu(torch.zeros((size_seta, size_seta)), config)
    for i in range(size_seta):
        index = i
        while index<size_seta:
            xsquare = torch.sum(X_a[i]**2, dim=0)
            ysquare = torch.sum(X_a[index:min(index+batch_size, size_seta)]**2, dim=1)
            xdoty = torch.sum(torch.mul(X_a[i], X_a[index:min(index+batch_size, size_seta)]), dim=1)
            dist = xsquare - 2*xdoty + ysquare

            pairwise_distances[i, index:min(index+batch_size, size_seta)] = dist
            pairwise_distances[index:min(index+batch_size, size_seta), i] = dist
            index += batch_size
        for k, kth in enumerate(config.kth_nearests):
            k_th_distance_i = np.partition(pairwise_distances[i].detach().cpu().numpy(), kth)[kth]
            k_th_distance[i, k] = float(k_th_distance_i)

    scores = np.zeros(len(config.kth_nearests))
    for i in range(size_setb):
        for k, kth in enumerate(config.kth_nearests):
            index = 0
            while index<size_seta:
                xsquare = convert_to_gpu(torch.sum(X_b[i]**2, dim=0), config)
                ysquare = convert_to_gpu(torch.sum(X_a[index:min(index+batch_size, size_seta)]**2, dim=1), config)
                xdoty = torch.sum(torch.mul(X_b[i], X_a[index:min(index+batch_size, size_seta)]), dim=1)
                dist = xsquare - 2*xdoty + ysquare
                differences = dist - k_th_distance[index:min(index+batch_size, size_seta), k]
                if torch.min(differences) <= 0 :
                    index = size_seta
                    scores[k] += 1
                else:
                    index += batch_size
    return scores/size_setb


def hausdorff_estimate(X_a, X_b, config):
    size_seta, size_setb, batch_size = X_a.shape[0], X_b.shape[0], config.batch_size_knn
    min_distance = np.zeros(size_seta)
    for i in range(size_seta):
        index = 0
        pairwise_distances = convert_to_gpu(torch.zeros((size_setb)), config)
        while index<size_setb:
            xsquare = torch.sum(X_a[i]**2, dim=0)
            ysquare = torch.sum(X_b[index:min(index+batch_size, size_setb)]**2, dim=1)
            xdoty = torch.sum(torch.mul(X_a[i], X_b[index:min(index+batch_size, size_setb)]), dim=1)
            dist = xsquare - 2*xdoty + ysquare
            pairwise_distances[index:min(index+batch_size, size_setb)] = dist
            index += batch_size
        min_distance[i] = np.amin(pairwise_distances.detach().cpu().numpy())
    max_of_mins = np.sqrt(np.mean(min_distance))
    return max_of_mins


def hausdorff_estimate_interpolation(X_a, X_b, metric_score, config):
    size_seta, size_setb, batch_size = X_a.shape[0], X_b.shape[0], config.batch_size_knn
    min_distance = np.zeros(size_seta)
    for i in range(size_seta):
        index = 0
        pairwise_distances = convert_to_gpu(torch.zeros((size_setb)), config)
        while index<size_setb:
            xsquare = torch.sum(X_a[i]**2, dim=0)
            ysquare = torch.sum(X_b[index:min(index+batch_size, size_setb)]**2, dim=1)
            xdoty = torch.sum(torch.mul(X_a[i], X_b[index:min(index+batch_size, size_setb)]), dim=1)
            dist = xsquare - 2*xdoty + ysquare
            pairwise_distances[index:min(index+batch_size, size_setb)] = dist
            index += batch_size
        min_distance[i] = np.amin(pairwise_distances.detach().cpu().numpy())
    plt.clf()
    plt.hist(min_distance, bins=10)
    if not os.path.exists(config.name_exp+'/distances_distribution'):
        os.makedirs(config.name_exp+'/distances_distribution')
    plt.savefig(config.name_exp+"/distances_distribution/"+str(config.num_pics)+"_"+metric_score, bbox_inches="tight")
    plt.savefig(config.name_exp+"/distances_distribution/"+str(config.num_pics)+"_"+metric_score+".pdf", bbox_inches="tight")
    max_of_mins = np.sqrt(np.mean(min_distance))
    return max_of_mins

def knn_scores(generator, config):
    z = generate_z(config.num_KNN, config.z_var, config)
    gz = generator(convert_to_gpu(z, config)).detach().cpu().numpy()
    z = z.detach().numpy()
    #1 Get the classes from (Gz) the position in the output space
    _, classes = calculate_distance_to_nearest_point(gz, config)

    #2 Get the mean of these classes in the latent space
    z_Kmeans = list()
    for this_class in np.unique(classes):
        indexes = np.where(classes==this_class)[0]
        z_this_class = z[indexes]
        z_mean = np.mean(z_this_class, axis=0)
        # if (config.z_dim + 1 >= config.output_modes):
        #     z_mean /= np.linalg.norm(z_mean)
        z_Kmeans.append(z_mean)
    z_Kmeans = np.array(z_Kmeans)
    distances_means = distance_matrix(z_Kmeans, z_Kmeans)
    distances_means = np.array([distances_means[i,j] for i in range(len(z_Kmeans)) for j in range(i+1, len(z_Kmeans))])
    simplicial_ratio = np.var(distances_means)/np.mean(distances_means)

    #3 Get the classes from (z) the position in the latent space
    _, classes_z_Kmeans = calculate_distance_to_nearest_point(z, config, real_data=z_Kmeans)

    #4 Compute the proportion of similarity in the classes !
    accuracies, lengths = list(), list()
    for this_class in np.unique(classes):
        indexes = np.where(classes_z_Kmeans==this_class)[0]
        classes_these_indexes = classes[indexes]
        if len(classes_these_indexes)>0:
            accuracies.append(np.amax(np.bincount(classes_these_indexes))/len(classes_these_indexes))
            lengths.append(len(classes_these_indexes))
    KNNacc = np.sum(np.multiply(np.array(accuracies), np.array(lengths)))/np.sum(np.array(lengths))
    return (KNNacc, simplicial_ratio)
