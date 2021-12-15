from my_imports import *


def find_MST(cities):
    matrix_distances = cdist(cities, cities, metric="euclidean")
    X = csr_matrix(matrix_distances)
    Tcsr = minimum_spanning_tree(X)
    MST = Tcsr.toarray().astype(float)
    points_to_match = list()
    for i in range(len(matrix_distances)):
        for j in range(len(matrix_distances)):
            if MST[i][j]>0:
                points_to_match.append((i,j))
    return np.array(points_to_match)

def find_MST_networkx(data):
    matrix_distances = cdist(data, data, metric="euclidean")
    G=nx.Graph()
    for i,j in product(range(len(matrix_distances)), repeat=2):
        G.add_edge(i, j, weight=matrix_distances[i][j])
    T = nx.minimum_spanning_tree(G)
    return T

def get_dataset_mnist(metric, config):
    if metric=="classifier":
        data, _ = config.classifier(config.real_dataset)
    elif metric=="L2":
        data = config.real_dataset.view((config.real_dataset.shape[0],-1))
    else:
        print("Not the right metric")
        sys.exit()
    data = data.detach().numpy()
    return data

def plot_MST(matches, cities, config):
    plt.clf()
    plt.scatter(cities[:,0],cities[:,1])
    for i in range(len(matches)):
        x = [cities[matches[:,0][i]][0], cities[matches[:,1][i]][0]]
        y = [cities[matches[:,0][i]][1], cities[matches[:,1][i]][1]]
        plt.plot(x, y, c="b")
    plt.axis('off')
    plt.savefig(config.name_exp+"/minimal_spanning_tree_"+str(config.output_modes), bbox_inches="tight")
    plt.savefig(config.name_exp+"/minimal_spanning_tree_"+str(config.output_modes)+".pdf", bbox_inches="tight")


path_distance_TSP = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
path_distance_quadratic = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]])**2 for p in range(len(r))])
path_distance_TSPvariant = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))
def two_opt(cities, improvement_threshold, path_distance): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route, cities) # and check the total distance with this modification.
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
    return route # When the route is no longer improving substantially, stop searching and return the route.

def plot_TSP(cities, config):
    route = two_opt(cities, 0.0001, path_distance_TSP)
    plt.clf()
    new_cities_order = np.concatenate((np.array([cities[route[i]] for i in range(len(route))]), np.array([cities[0]])))
    plt.scatter(cities[:,0], cities[:,1])
    plt.plot(new_cities_order[:,0], new_cities_order[:,1], c="b")
    plt.axis('off')
    plt.savefig(config.name_exp+"/shortest_path_"+str(config.output_modes), bbox_inches="tight")
    plt.savefig(config.name_exp+"/shortest_path_"+str(config.output_modes)+".pdf", bbox_inches="tight")

def plot_TSP_notclosed(cities, config):
    route = two_opt(cities, 0.0001, path_distance_TSPvariant)
    plt.clf()
    new_cities_order = np.array([cities[route[i]] for i in range(len(route))])
    plt.scatter(cities[:,0], cities[:,1])
    plt.plot(new_cities_order[:,0], new_cities_order[:,1], c="b")
    plt.axis('off')
    plt.savefig(config.name_exp+"/shortest_path_variant_"+str(config.output_modes), bbox_inches="tight")
    plt.savefig(config.name_exp+"/shortest_path_variant_"+str(config.output_modes)+".pdf", bbox_inches="tight")

def plot_TSP_quadratic(cities, config):
    route = two_opt(cities, 0.0001, path_distance_quadratic)
    plt.clf()
    new_cities_order = np.array([cities[route[i]] for i in range(len(route))])
    plt.scatter(cities[:,0], cities[:,1])
    plt.plot(new_cities_order[:,0], new_cities_order[:,1], c="b")
    plt.axis('off')
    plt.savefig(config.name_exp+"/shortest_path_quadratic_"+str(config.output_modes), bbox_inches="tight")
    plt.savefig(config.name_exp+"/shortest_path_quadratic_"+str(config.output_modes)+".pdf", bbox_inches="tight")

def plot_MST_networkx(metric, config):
    data = get_dataset_mnist(metric, config)
    T = find_MST_networkx(data)
    plt.clf()
    nx.draw(T, with_labels=False, font_weight='bold')
    pos = nx.spring_layout(T)
    labels = {}    
    for node in list(T.nodes())[:config.real_dataset_size]:
        labels[node] = config.real_dataset_labels[node]
    nx.draw_networkx_labels(T, pos, labels, font_size=20, font_color='k')
    plt.axis('off')
    filename = "/mnist"+str(config.real_dataset_size)+"_"+metric+"_MST"
    plt.savefig(config.name_exp+"/MST_networkx_"+metric+"_"+str(config.real_dataset_size)+"samples", bbox_inches="tight")
    plt.savefig(config.name_exp+"/MST_networkx_"+metric+'_'+str(config.real_dataset_size)+"samples.pdf", bbox_inches="tight")
    

def plot_MST_mnist(metric, config):
    data = get_dataset_mnist(metric, config)
    cities = get_projections_mnist(data)
    matches = find_MST(data)
    plt.clf()
    plt.scatter(cities[:,0],cities[:,1])
    for i in range(len(matches)):
        x = [cities[matches[:,0][i]][0], cities[matches[:,1][i]][0]]
        y = [cities[matches[:,0][i]][1], cities[matches[:,1][i]][1]]
        plt.plot(x, y, c="b")
    for label, x, y in zip(config.real_dataset_labels, cities[:, 0], cities[:, 1]):
        plt.annotate(str(label)[7:-1], xy = (x, y), xytext = (-20, 20), fontsize=15,
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.axis('off')
    filename = "/mnist"+str(config.real_dataset_size)+"_"+metric+"_MST"
    plt.savefig(config.name_exp+"/minimal_spanning_tree_"+metric+"_"+str(config.real_dataset_size)+"samples", bbox_inches="tight")
    plt.savefig(config.name_exp+"/minimal_spanning_tree_"+metric+'_'+str(config.real_dataset_size)+"samples.pdf", bbox_inches="tight")
    return matches


def get_projections_mnist(data):
    data = cdist(data, data, metric="euclidean")
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=123, n_init=10, n_jobs=-1, max_iter=500)
    results = mds.fit(data)
    coords = results.embedding_
    return coords