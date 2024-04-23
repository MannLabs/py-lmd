
import numpy as np
from numba import njit, types, prange
import numba as nb
import numpy as np

from hilbertcurve.hilbertcurve import HilbertCurve

import numba as nb

@njit
def _create_coord_index(mask, background=0):

    num_classes = np.max(mask)+1

    # each class will have a list of coordinates
    # each coordinate is a 2D array with row and column
    # initial size is 32, whenever it exceeds the size, it will be doubled
    initial_size = 32

    index_list = []
    stop_list = np.zeros(num_classes, dtype=np.uint32)
    for i in range(num_classes):
        index_list.append(np.zeros((initial_size, 2), dtype=np.uint32))

    # create index list
    for col in range(mask.shape[1]):
        for row in range(mask.shape[0]):
            class_id = mask[row, col]
            if class_id != background:
                if stop_list[class_id] >= index_list[class_id].shape[0]:
                    new_size = index_list[class_id].shape[0] * 2
                    new_array = np.zeros((new_size, 2), dtype=np.uint32)
                    new_array[:stop_list[class_id]] = index_list[class_id]
                    index_list[class_id] = new_array
                index_list[class_id][stop_list[class_id]][0] = row
                index_list[class_id][stop_list[class_id]][1] = col
                stop_list[class_id] += 1

    # resize index list
    for i in range(num_classes):
        index_list[i] = index_list[i][:stop_list[i]]

    return index_list

@njit
def _filter_coord_index(index_list, classes):

    filtered_index_list = []
    for idx, class_id in enumerate(classes):
        filtered_index_list.append(index_list[class_id])
    return filtered_index_list
             
def get_coordinate_form(inarr, classes, coords_lookup, debug=False):
    # return with empty lists if no classes are provided
    if len(classes) == 0:
        return [],[],[]
    
    coords_filtered = _filter_coord_index(coords_lookup, nb.typed.List(classes))
    
    center = [np.mean(el, axis=0) for el in coords_filtered]
    
    if debug:
        print("start length calculation")
    
    length = [len(el) for el in coords_filtered]
    
    return center, length, coords_filtered

def tsp_hilbert_solve(data , p=3):

    p=p; n=2
    max_n = 2**(p*n)
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(max_n))
    hilbert_points = hilbert_curve.points_from_distances(distances)
    hilbert_points = np.array(hilbert_points)

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    hilbert_min = np.min(hilbert_points, axis=0)
    hilbert_max = np.max(hilbert_points, axis=0)

    data_scaled = data - data_min
    data_scaled = data_scaled / (data_max-data_min) * (hilbert_max - hilbert_min)

    data_rounded = np.round(data_scaled).astype(int)

    order = assign_vertices(hilbert_points, data_rounded)
    
    return order

# return the first element not present in a list
def _get_closest(used, choices, world_size):
    for element in choices:
        if element not in used:
            # knn matrix contains -1 if the number of elements is smaller than k
            if element == -1:
                return None
            else:
                return element
        
    return None
    # all choices have been taken, return closest free index due to local optimality
    
def _tps_greedy_solve(data, k=100):
    samples = len(data)
    
    print(f"{samples} nodes left")
    #recursive abort
    if samples == 1:
        return data
    
    import umap
    knn_index, knn_dist, _ = umap.umap_.nearest_neighbors(data, n_neighbors=k, 
                                       metric='euclidean', metric_kwds={},
                                       angular=True, random_state=np.random.RandomState(42))

    knn_index = knn_index[:,1:]
    knn_dist = knn_dist[:,1:]

    # follow greedy knn as long as a nearest neighbour is found in the current tree            
    nodes = []
    current_node = 0
    while current_node is not None:
        nodes.append(current_node)
        #print(current_node, knn_index[current_node], next_node)
        next_node = _get_closest(nodes, knn_index[current_node], samples)
        
        current_node = next_node

    # as soon as no nearest neigbour can be found, create a new list of all elements still remeining
    # nodes: [0, 2, 5], nodes_left: [1, 3, 4, 6, 7, 8, 9]
    # add the last node assigned as starting point to the new list
    # nodes: [0, 2], nodes_left: [5, 1, 3, 4, 6, 7, 8, 9]

  
    nodes_left = list(set(range(samples))-set(nodes))
    

    # add last node from nodes to nodes_left

    nodes_left = [nodes.pop(-1)] + nodes_left
    

    node_data_left = data[nodes_left]
    
    # join lists
    
    return np.concatenate([data[nodes], _tps_greedy_solve(node_data_left, k=k)])

# calculate the index array for a sorted 2d list based on an unsorted list
@njit()
def _get_nodes(data, sorted_data):
    indexed_data = [(i,el) for i, el in enumerate(data)]

    epsilon = 1e-10
    nodes = []

    print("start sorting")
    for element in sorted_data:

        for j, tup in enumerate(indexed_data):
            i, el = tup

            if np.array_equal(el, element):
                nodes.append(i)
                indexed_data.pop(j)
    return nodes
    
def tsp_greedy_solve(node_list, k=100, return_sorted=False):
    """Find an approximation of the closest path through a list of coordinates
    
    Args:
        node_list (np.array): Array of shape `(N, 2)` containing a list of coordinates
        
        k (int, default: 100): Number of Nearest Neighbours calculated for each Node.
        
        return_sorted: If set to False a list of indices is returned. If set to True the sorted coordinates are returned.
    
    """
    
    sorted_nodes = _tps_greedy_solve(node_list)
    
    if return_sorted:
        return sorted_nodes
        
    else:
        nodes_order = _get_nodes(node_list, sorted_nodes)
        return nodes_order
    
@njit()
def assign_vertices(hilbert_points, data_rounded):

    data_rounded = data_rounded.astype(np.int64)
    hilbert_points = hilbert_points.astype(np.int64)


    output_order = np.zeros(len(data_rounded)).astype(np.int64)
    current_index = 0

    for hilbert_point in hilbert_points:

        for i, data_point in enumerate(data_rounded):
            if np.array_equal(hilbert_point, data_point):
                output_order[current_index] = i
                current_index += 1

    return output_order
    
def calc_len(data):
    """calculate the length of a path based on a list of coordinates
    
    Args:
        data (np.array): Array of shape `(N, 2)` containing a list of coordinates
       
    """

    index = np.arange(len(data)).astype(int)
    
    not_shifted = data[index[:-1]]
    shifted = data[index[1:]]
    
    diff = not_shifted-shifted
    sq = np.square(diff)
    dist = np.sum(np.sqrt(np.sum(sq, axis = 1)))
    
    return dist