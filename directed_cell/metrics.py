import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import pairwise as skpair
from directed_cell.utils import get_graph, get_numpy, get_scipy
from directed_cell.embeddings import Embedder
from itertools import combinations

"""
    0   1   2
   -----------
0 | 0   0   1
1 | 1   0   1
2 | 0   1   0

# A_from_to (row is from, col is to)

"""

mappings = {
    'max' : np.max,
    'min' : np.min,
    'mean' : np.mean,
    'none' : lambda x : x,
    'prc_95' : lambda x : np.percentile(x, 0.95),
    'prc_05' : lambda x : np.percentile(x, 0.05)
}


def in_degree(A, type='max'):
    assert type in mappings.keys(), "not valid aggregation type"
    A = get_numpy(A)
    deg = np.sum(A, axis=1)
    return mappings[type](deg)

def out_degree(A, type='max'):
    assert type in mappings.keys(), "not valid aggregation type"
    A = get_numpy(A)
    deg = np.sum(A, axis=0).T
    return mappings[type](deg)

def assortativity(A, weight=None):
    """
    https://en.wikipedia.org/wiki/Assortativity
    """
    if weight is not None:
        # Have to be integers
        A = get_scipy(A).astype(int)

    return nx.degree_assortativity_coefficient(get_graph(A), weight=weight)

def power_law_exponent(A, type='in'):
    degree_dist = in_degree(A, type='none') if type =='in' else out_degree(A, type='none')
    degree_dist = degree_dist[degree_dist != 0]
    #degree_dist = degree_dist/degree_dist.sum()
    n = 1000 # Constant
    min_degree = max(np.min(degree_dist), 1)
    return 1 + n/np.sum((np.log(degree_dist/min_degree)))

def relative_edge_dist_entropy(A, type='in'):
    """
    0 means the degree distribution is uniform,
    1 means one node is connected to all others
    """
    A = get_numpy(A)
    degree_dist = in_degree(A, type='none') if type =='in' else out_degree(A, type='none')
    #degree_dist = degree_dist[degree_dist != 0]
    degree_dist = degree_dist/degree_dist.sum()
    m = np.count_nonzero(A) # edge count
    N = A.shape[0] # node count
    RDE = -1/np.log(N) * (np.sum(degree_dist)/(2*m))*np.log(np.sum(degree_dist)/(2*m))
    return RDE

def average_local_clustering_coefficient(A):
    return nx.cluster.average_clustering(get_graph(A))
    
def global_clustering_coefficient(A):
    return nx.cluster.clustering(get_graph(A))

def weighted_clustering_coefficient(A):
    A = get_graph(A.copy())
    A = nx.convert_node_labels_to_integers(A)
    local_clusterings = []
    for i in A.nodes:
        out_edge_data = A.out_edges(i, data=True)
        neighbors = [v for _, v, _ in out_edge_data]
        edge_weights = {(i, j) : data['weight'] for i, j, data in out_edge_data}
        node_strength = sum(edge_weights.values())
        n_neighbors = len(edge_weights)
        local_clustering = 0
        
        for j in neighbors:
            for h in neighbors:
                if A.has_edge(j, h):
                    local_clustering += (edge_weights[(i, j)] + edge_weights[(i, h)])/2
        if local_clustering != 0 and n_neighbors != 1 and node_strength != 0:
            local_clustering *= 1/(node_strength*(n_neighbors-1))
        local_clusterings.append(local_clustering)
    
    return sum(local_clusterings)/len(local_clusterings)

    
def characteristic_path_length(A):
    """
    Returns the averagae shortest (characteristic) path length for a graph A
    https://en.wikipedia.org/wiki/Average_path_length
    """
    A = get_graph(A)
    N = len(A.nodes)
    s = 0
    for node, path_length_dict in nx.shortest_path_length(A):
        s += sum(path_length_dict.values())

    return s/(N * (N - 1))

def directed_wedge_count(G, directed = True):
    G = get_graph(G)
    assert G.is_directed()
    count = 0
    G = get_graph(G)
    for n in G.nodes:
        count += len(list(combinations(G.out_edges(n), 2)))
            
    return int(count)

def directed_triangle_count(G):
    G = get_graph(G)
    assert G.is_directed()
    count = 0
    for n1 in G.nodes:
        out_neighs_1 = [n[1] for n in G.out_edges(n1) if n[0] != n[1]]
        for n2 in out_neighs_1:
            out_neighs_2 = [n[1] for n in G.out_edges(n2) if n[0] != n[1]]
            count += len([n3 for n3 in out_neighs_2 if G.has_edge(n3, n1) and n3 != n1])
    
    return int(count/3)

def directed_square_count(G):
    G = get_graph(G)
    assert G.is_directed()
    count = 0
    for n1 in G.nodes:
        out_neighs_1 = [n[1] for n in G.out_edges(n1) if n[0] != n[1]]
        for n2 in out_neighs_1:
            out_neighs_2 = [n[1] for n in G.out_edges(n2) if n[0] != n[1]]
            for n3 in out_neighs_2:
                out_neighs_3 = [n[1] for n in G.out_edges(n3) if n[0] != n[1]]
                count += len([n4 for n4 in out_neighs_3 if G.has_edge(n4, n1) and n4 != n1])
    
    return int(count/4)

def undirected_wedge_count(G):
    """
    Compute the wedge count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Wedge count.
    """
    G = get_scipy(get_graph(G).to_undirected())
    degrees = np.array(G.sum(axis=-1))
    return 0.5 * np.dot(degrees.T, degrees - 1).reshape([])

def undirected_triangle_count(G):
    """
    Compute the triangle count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Triangle count.
    """
    G_graph = get_graph(G).to_undirected()
    triangles = nx.triangles(G_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def undirected_square_count(G):
    """
    Compute the square count.
    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
    
    Returns:
        Square count.
    """
    G = get_scipy(get_graph(G).to_undirected())
    A_squared = G @ G
    common_neighbors = sp.triu(A_squared, k=1).tocsr()
    num_common_neighbors = np.array(
        common_neighbors[common_neighbors.nonzero()]
    ).reshape(-1)
    return np.dot(num_common_neighbors, num_common_neighbors - 1) / 4

def diameter(G):
    """Computes longest shortest path"""
    G = get_graph(G)
    s = set()
    for node, path_length_dict in nx.shortest_path_length(G):
        s.update(list(path_length_dict.values()))
        

    return max(s)

def gini_coef(A, type='in'):
    A = get_numpy(A)
    degree_dist = in_degree(A, type='none') if type =='in' else out_degree(A, type='none')
    degree_dist = np.sort(np.array(degree_dist).squeeze())
    N = A.shape[0]
    gini = (2*np.sum(degree_dist * np.arange(len(degree_dist)))) / \
        (N*np.sum(degree_dist)) - (1 + 1/N)

    return gini

def edge_distribution_entropy(A, type='in'):
    A = get_numpy(A)
    degrees = in_degree(A, type='none') if type =='in' else out_degree(A, type='none')
    m = 0.5 * np.sum(np.square(A))
    n = A.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er

def MMD(X: np.matrix, Y: np.matrix, kernel=skpair.rbf_kernel, **kernel_args):
    """Empirical MMD. The closer the value is to 0, the more equal the empirical distributions"""
    m = X.shape[0]
    n = Y.shape[0]
    XX = np.sum(kernel(X, X, **kernel_args))
    YY = np.sum(kernel(Y, Y, **kernel_args))
    XY = np.sum(kernel(X, Y, **kernel_args))

    return 1/(m**2) * XX - 2/(m*n) * XY + 1/(n**2) * YY

def largest_scc(A):
    return len(sorted(nx.strongly_connected_components(get_graph(A)), key=len)[-1])


def node_embedding_mmd(embedder: Embedder, kernel:callable, G: nx.DiGraph, G_hat: nx.DiGraph):
    """Compute MMD between two graphs given an embedder and a kernel"""
    G_embed, G_hat_embed = embedder.embed_multigraph([G, G_hat])
    return MMD(G_embed, G_hat_embed, kernel)
