from my_imports import *
import os, sys

def convert_to_gpu(data, config):
    if config.use_gpu:
        return data.cuda(config.device_id)
    else:
        return data

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        print(Pr[i, k])
        path.append(Pr[i, k])
        k = Pr[i, k]
    returned_path = path[::-1]
    print(len(returned_path))
    return returned_path

def get_all_interpolations_MST(MST, config):
    num_inter = config.num_points_sampled_knn//config.real_dataset_size
    lambdas = np.linspace(0,1,num=num_inter)
    interpolationMST = list()
    for elem in MST:
        interpolationMST += [lam*config.real_dataset[elem[0]]+(1-lam)*config.real_dataset[elem[1]] for lam in lambdas]
    result = torch.cat(interpolationMST)
    return result

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.5)
        #nn.init.zero_(m.weight)
        #nn.init.eye_(m.weight)
        #nn.init.orthogonal_(m.weight)

def read_results(folder):
    list_files = os.listdir(folder)
    results = dict()
    for file in list_files:
        if file.endswith(".txt"):
            f = open (os.path.join(folder, file), "r")
            data = json.loads(f.read())
            precision = max([data[elem]['prec'][0] for elem in data.keys()])
            knn_acc = max([data[elem]['knn'][0] for elem in data.keys()])
            simplicial_ratio = max([data[elem]['knn'][1] for elem in data.keys()])
            results[file] = [precision, knn_acc, simplicial_ratio]

    print(results)
    sys.exit()
