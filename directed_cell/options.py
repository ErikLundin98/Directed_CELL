from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx

from directed_cell import utils

# Loss functions


class LossFunction(ABC):
    """Base class for all loss functions"""
    @abstractmethod
    def __call__(self, W, A, n_edges):
        raise NotImplementedError

class RegularLossFunction(LossFunction):
    """Regular Cross-Entropy with logits loss function"""

    def __call__(self, W, A, n_edges):
        """Standard loss function for directed graphs"""
        self.d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        return 0.5 * torch.sum(A * (self.d - W)) / n_edges

class LocalityLossFunction(RegularLossFunction):
    """Enforces short edges based on original graph"""
    def __init__(self, A, max_len, c=0.5):
        super().__init__()
        G = nx.DiGraph(A)
        A = nx.to_numpy_matrix(G)
        D = np.full(A.shape, np.inf)
        
        shortest_paths = nx.shortest_path_length(G)
        for (n, path_dict) in shortest_paths:
            D[n, list(path_dict.keys())] = list(path_dict.values())
        
        D[D>max_len] = 0
        D[D<=max_len] = 1
        loss_filter = np.multiply(A, D) 
        self.loss_filter = (torch.tensor(loss_filter)).clone().detach()
        self.c = c

    def __call__(self, W, A, n_edges):
        l1 = super().__call__(W, A, n_edges)
        l2 = 0.5 * torch.sum(self.loss_filter * (self.d - W)) / n_edges
        return self.c*l1 + (1-self.c)*l2

class LazyLossFunction(LossFunction):
    """Who needs random walks anyways"""
    def __init__(self):
        super().__init__()

    def __call__(self, W, A, n_edges):
        e = torch.exp(W)
        p = e / torch.sum(e)
        
        return - torch.sum(A*torch.log(p))/n_edges

# Sampling functions

class SamplingFunction(ABC):
    """Base class for all sampling functions"""
    def __init__(self, verbose=False):
        self.require_strong_connectivity = True
        self.verbose = verbose
    
    @abstractmethod
    def __call__(self, model, n_edges=None, self_loops=False):
        raise NotImplementedError


class SampleGraphRegular(SamplingFunction):
    """Sample edges from the score matrix without replacement"""
    def __init__(self, verbose=False):
        super().__init__(verbose)
        

    def __call__(self, model, n_edges=None, self_loops=False):
        """Generate graph with desired edge count"""
        S = model.get_scores_matrix()
        G = sp.lil_matrix(S.shape, dtype=int)
        N = S.shape[0]
        
        if not n_edges: 
            n_edges = model.n_edges

        if not self_loops: np.fill_diagonal(S, 0)
        S = S/S.sum()
        n_possible = np.count_nonzero(S)
        if n_possible < n_edges: print(f"warning: cannot sample enough edges ({n_possible, n_edges})")
        
        target = np.random.choice(np.arange(N*N), size=int(min(n_edges, n_possible)), p = S.ravel(), replace=False)
        target_indices = np.unravel_index(target, (N, N))
        G[target_indices] = 1

        if self.verbose: print("generated edges:", G.count_nonzero(), ", desired:", n_edges)
        return G


class SampleGraphCELL(SamplingFunction):
    """Sample edges from the score matrix node-wise with replacement, then from the whole matrix without replacement"""
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def __call__(self, model, n_edges=None, self_loops=False):
        """
        Generate graph using the same procedure as in the original undirected CELL paper.
        """
        S = model.get_scores_matrix()
        G = sp.lil_matrix(S.shape, dtype=int)
        N = S.shape[0]
        if not self_loops: np.fill_diagonal(S, 0)
        d = S.sum(axis=1)

        if not n_edges: 
            n_edges = model.n_edges

        for n in range(N):  # Iterate over the nodes
            p=S[n] / d[n]
            target = np.random.choice(N, p=p, replace=True)
            G[n, target] = 1

        diff = np.round(n_edges - G.sum())
        if diff > 0:
            if self.verbose: print("sampling additional edges:", diff)
            S[G.nonzero()] = 0
            S = S/S.sum()

            target = np.random.choice(np.arange(N*N), size=int(diff), p = S.ravel(), replace=False)
            target_indices = np.unravel_index(target, (N, N))
            G[target_indices] = 1

        if self.verbose: print("generated edges:", G.count_nonzero(), ", desired:", n_edges)

        return G

class SampleGraphFromPaths(SamplingFunction):
    def __init__(self, path_len=15, verbose=False):
        super().__init__(verbose)
        self.path_len=path_len
        pass

    def __call__(self, model, n_edges=None, self_loops=False):
        """Generate graph with desired edge count"""
        W = model._get_W()
        P = model._get_synthetic_P(W)
        pi = model._get_pi(P)
        if np.any(pi < 0) and not np.all(pi < 0):
            pi = np.abs(pi)
            pi = pi/pi.sum()
        if not self_loops:
            np.fill_diagonal(P, 0)
        
        P = P/P.sum(axis=1, keepdims=1)
        G = sp.lil_matrix(W.shape, dtype=int)
        N = W.shape[0]
        
        if not n_edges: 
            n_edges = model.n_edges

        current_n_edges = 0
        while True:
            current_node = np.random.choice(a=N, p=pi)
            for i in range(self.path_len):
                next_node = np.random.choice(a=N, p=P[current_node])
                if G[current_node, next_node] == 0:
                    current_n_edges += 1
                    G[current_node, next_node] = 1

                current_node = next_node

                if current_n_edges >= n_edges: break
            if current_n_edges >= n_edges: break

        if self.verbose: print("generated edges:", G.count_nonzero(), ", desired:", n_edges)
        return G


class SampleGraphLazy(SamplingFunction):
    """Who needs random walks anyways?"""
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.require_strong_connectivity = False
        

    def __call__(self, model, n_edges=None, self_loops=False):
        """Generate graph with desired edge count"""
        W = model._get_W().detach().numpy()
        e = np.exp(W)
        S = e/np.sum(e)
        N = S.shape[0]
        G = sp.lil_matrix(shape=W.shape)
        
        if not n_edges: 
            n_edges = model.n_edges

        if not self_loops: np.fill_diagonal(S, 0)
        S = S/S.sum()
        n_possible = np.count_nonzero(S)
        if n_possible < n_edges: print(f"warning: cannot sample enough edges ({n_possible, n_edges})")
        
        target = np.random.choice(np.arange(N*N), size=int(min(n_edges, n_possible)), p = S.ravel(), replace=False)
        target_indices = np.unravel_index(target, (N, N))
        G[target_indices] = 1

        if self.verbose: print("generated edges:", G.count_nonzero(), ", desired:", n_edges)
        return G


class SampleGraphUndirectedCELL(SamplingFunction):
    """Sample edges from the score matrix node-wise with replacement, then from the whole matrix without replacement"""
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def __call__(self, model, n_edges=None, self_loops=False):
        """
        Generate graph using the same procedure as in the original undirected CELL paper.
        """
        scores_matrix = model.get_scores_matrix()
        scores_matrix = scores_matrix + scores_matrix.T
        target_g = sp.lil_matrix(scores_matrix.shape, dtype=int)

        if not n_edges: 
            n_edges = model.n_edges

        if not self_loops:
            np.fill_diagonal(scores_matrix, 0)

        degrees = scores_matrix.sum(1)  # The row sum over the scores_matrix.

        N = scores_matrix.shape[0]
        for n in range(N):  # Iterate over the nodes
            target = np.random.choice(N, p=scores_matrix[n] / degrees[n])
            target_g[n, target] = 1
            target_g[target, n] = 1

        diff = np.round((2 * n_edges - target_g.sum()) / 2)
        if diff > 0:
            triu = np.triu(scores_matrix)
            triu[target_g.nonzero()] = 0
            triu = triu / triu.sum()
            
            n_possible = np.count_nonzero(triu)

            triu_ixs = np.triu_indices_from(scores_matrix)
            extra_edges = np.random.choice(
                triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=min(int(diff), int(n_possible))
            )

            target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
            target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1
        if self.verbose:
            print("generated edges:", target_g.sum()/2, ", desired:", n_edges)
        return sp.csr_matrix(target_g)

# Early stopping criteria


class Criterion(ABC):
    """Base class for all early stopping criteria"""
    def __init__(self, interval:int=1, verbose=False):
        self.interval = interval
        self.verbose = verbose

    @abstractmethod
    def __call__(self, i, A, model):
        raise NotImplementedError

class EdgeOverlapCriterion(Criterion):
    """Callable class for determining when edge overlap is large enough to stop training"""
    def __init__(self, A=None, interval:int=1, overlap:float=0.52, verbose=False, directed=True):
        super().__init__(interval, verbose)
        self.A = A
        self.overlap = overlap
        self.directed = directed


    def __call__(self, i, model):
        if i % self.interval == 0:
            A_hat = model.sample_graph()
            if self.A is None:
                A = model.A
            else:
                A = self.A
            overlap = utils.edge_overlap(A, A_hat, directed=self.directed)
            if self.verbose: print("overlap:", overlap)
            return overlap >= self.overlap
        else:
            return False

