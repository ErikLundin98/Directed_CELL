import networkx as nx
import numpy as np
from random import choice
import scipy.stats as stats
import scipy.sparse as sp
from directed_cell.utils import get_scipy, is_weighted


def sample_configuration_digraph(G: nx.DiGraph):
    """
    Sample a Directed Graph using the Directed Configuration model.
    The graph will be constructed to have the same degree sequence as G, but self loops and MultiGraph edges are removed, so it will not be exactly the same
    """
    in_degrees = list(d for (_, d) in G.in_degree())
    out_degrees = list(d for (_, d) in G.out_degree())

    G = nx.directed_configuration_model(in_degrees, out_degrees, create_using=nx.DiGraph)
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def sample_erdos_renyi(G: nx.Graph):
    """
    Sample a (Un)Directed Graph using the Erdos Renyi model. The graph will have approximately as many edges as G
    """
    directed = G.is_directed()
    N = len(G.nodes)
    E = len(G.edges)
    return nx.erdos_renyi_graph(n=len(G.nodes), p=E/(N*N), directed=directed)


def randomize_edge_directions(G_hat: nx.Graph, G: nx.DiGraph):
    """Randomize edge directions of an undirected graph G_hat based on the distribution of 'edge directions' in G"""
    assert not G_hat.is_directed()
    assert G.is_directed()
    
    # determine prior distribution of bidirectional edges and monodirectional edges
    A = nx.to_scipy_sparse_matrix(G)
    num_edges = len(G.edges)
    num_bidirected = A.multiply(A.T).count_nonzero()
    
    num_monodirected = len(G.edges) - num_bidirected
    
    # in an undirected graph, the probabilities are given by
    num_undirected = num_edges - num_bidirected/2
    p_bidirected = (num_bidirected/2) / num_edges
    
    G_hat_directed = G_hat.to_directed()
    
    for u, v in G_hat.edges:
        if np.random.choice([True, False], p=[num_bidirected/num_edges, num_monodirected/num_edges]):
            # bidirected edge
            continue
        elif choice([True, False]):
            # monodirected, from u to v
            G_hat_directed.remove_edge(v, u)
        else:
            # monodirected, from v to u
            G_hat_directed.remove_edge(u, v)
    
    return G_hat_directed


def sample_weights_from_normal_distribution(G_hat:nx.DiGraph, G:nx.DiGraph):
    """Fit a normal distribution to G's edge weight distribution and sample from it to add weights to G_hat"""
    assert is_weighted(G)
    A_hat = sp.lil_matrix(get_scipy(G_hat))
    weight_distribution = [data['weight'] for u, v, data in G.edges.data()]
    mu, sigma = stats.distributions.norm.fit(weight_distribution)
    edge_idx = A_hat.nonzero()

    preds = np.random.normal(mu, sigma, len(G.edges))
    preds[preds < 0] = 0
    A_hat[edge_idx] = preds
    return nx.DiGraph(A_hat)


def sample_weights_with_replacement(G_hat:nx.DiGraph, G:nx.DiGraph):
    """Sample edge weights with replacement from G and add them to G_hat"""
    assert is_weighted(G)
    A_hat = sp.lil_matrix(get_scipy(G_hat))
    weight_distribution = [data['weight'] for u, v, data in G.edges.data()]
    weights = np.random.choice(weight_distribution, size=len(G.edges))
    edge_idx = A_hat.nonzero()

    A_hat[edge_idx] = weights
    return nx.DiGraph(A_hat)