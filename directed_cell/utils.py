import scipy.sparse as sp
import numpy as np
import networkx as nx

from directed_cell.strong_connectivity import STRONGCONNECT, augment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_graph(A):
    if isinstance(A, np.ndarray):
        return nx.DiGraph(A)
    elif isinstance(A, sp.spmatrix):
        return nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    elif isinstance(A, nx.Graph):
        return A
    return None

def get_numpy(G):
    if isinstance(G, np.ndarray):
        return G
    elif isinstance(G, sp.spmatrix):
        return G.toarray()
    elif isinstance(G, nx.Graph):
        return nx.to_numpy_matrix(G)
    return None

def get_scipy(G):
    if isinstance(G, sp.spmatrix):
        return G
    elif isinstance(G, np.ndarray):
        return sp.csr_matrix(G)
    elif isinstance(G, nx.Graph):
        return sp.csr_matrix(get_numpy(G))
    return None

def is_weighted(G):
    A = get_numpy(G)
    return not np.array_equal(A, A.astype(bool))

def to_unweighted(G:nx.Graph):
    A = nx.to_numpy_matrix(G.copy())
    A[np.nonzero(A)] = 1
    return nx.DiGraph(A) if G.is_directed() else nx.Graph(A)

def edge_overlap(A, B, directed=True):
    """Edge overlap in % shared edges"""
    denominator = 1 if directed else 2
    n_edges = A.sum()/denominator
    if isinstance(A, np.ndarray): # assumes that B is of the same class
        overlap = np.logical_and(A, B).sum() / (denominator*n_edges)
    elif isinstance(A, sp.csr_matrix) or isinstance(A, sp.lil_matrix):
        overlap = A.multiply(B).sum() / (denominator*n_edges)
    else:
        raise TypeError("A and B must be of type numpy.ndarray or scipy.sparse.csr_matrix | scipy.sparse.lil_matrix")
    return overlap

def make_edges_less_probable(A: sp.csr_matrix, E:list, denominator:float=10.0):
    for U, V in E:
        A[U, V] = A[U, V]/denominator

    return sp.csr_matrix(A)



def graph_augment_strongly_connected(G:nx.DiGraph, denominator=10.0):
    """Augment a graph by making it strongly connected and dividing augmented edges with denominator"""
    if not nx.is_strongly_connected(G):
        E_sc = STRONGCONNECT(G)
        G = augment(G, E_sc)
    
        G = make_edges_less_probable(nx.to_scipy_sparse_matrix(G, format='lil'), E_sc, denominator=denominator)
    else:
        G = nx.to_scipy_sparse_matrix(G)
    return G

def graph_augment_weakly_connected(G:nx.Graph, denominator=10.0):
    """Augment a graph by making it weakly connected and dividing augmented edges with denominator"""
    if not nx.is_connected(G):
        E_c = list(nx.k_edge_augmentation(G, k=1))
        G.add_edges_from(E_c)
        G = make_edges_less_probable(nx.to_numpy_matrix(G), E_c, denominator=denominator)
    else:
        G = nx.to_scipy_sparse_matrix(G)
    return G

def load_dataset(path, format=['from', 'to'], directed=True, delim=',', comments='#') -> nx.Graph:
    """Loads a graph from a file and preprocesses it"""
    if 'weight' in format:
        weighted = True
        weight_idx = format.index('weight')
    else:
        weighted = False

    from_idx = format.index('from')
    to_idx = format.index('to')

    with open(path, 'r') as f:
        lines = f.readlines()
        res = [l.rstrip().split(delim) for l in lines if comments not in l]
        for i, l in enumerate(res):
            new_row = []
            new_row.append(l[from_idx])
            new_row.append(l[to_idx])
            if weighted:
                new_row.append({'weight': float(l[weight_idx])})

            res[i] = new_row
        
        G = nx.DiGraph() if directed else nx.Graph()
        #G = nx.parse_edgelist(res, comments=comments, delimiter=delim, create_using=create_using)
        G.add_edges_from(res)
        G.remove_edges_from(nx.selfloop_edges(G))
        return nx.convert_node_labels_to_integers(G) # reset node ordering


def graph_summary(G:nx.DiGraph):
    return {
        "|N|": len(G.nodes),
        "|E|": len(G.edges),
        "#Nodes in largest SCC" : sorted(len(sc) for sc in nx.strongly_connected_components(G))[-1]
    }