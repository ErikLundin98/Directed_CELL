"""
This file contains the code for the CELL model. It requires a sampling function, loss function and can optionally take in an early stopping criterion - all of which are found in options.py
"""

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import numpy as np
import torch
import networkx as nx
from directed_cell.utils import graph_augment_strongly_connected, graph_augment_weakly_connected

from directed_cell.options import LossFunction, SamplingFunction, Criterion

DEVICE = 'cpu'
DTYPE = torch.float32

class CELL:
    """Cross Entropy Logits with Low-Rank model"""
    def __init__(
        self,
        A : sp.csr_matrix,
        loss_fn : LossFunction,
        sampling_fn : SamplingFunction,
        criterion : Criterion,
        H : int = 9,
        n_edges: int = None,
        directed = True,
        augmentation_denominator = 10
    ):
        """
        A : Input adjacency matrix. Should be in a binary format
        loss_fn : 
        """
        self.A = A
        self.N = A.shape[0]

        self.n_edges = n_edges if n_edges else A.count_nonzero()

        self.H = H
        self.gamma = np.sqrt(2/(self.N+self.H)) # "normalization" for weight initialization
        self.augmentation_denominator = augmentation_denominator
        self.directed = directed

        if self.directed and sampling_fn.require_strong_connectivity:
            print("performing strong connectivity augmentation...")
            self.A = graph_augment_strongly_connected(self.construct_graph(self.A), self.augmentation_denominator) 
            assert nx.algorithms.is_strongly_connected(self.construct_graph(self.A)), 'Graph must be strongly connected!'
        elif not self.directed:
            print("performing weak connectivity augmentation...")
            self.A = graph_augment_weakly_connected(self.construct_graph(self.A), self.augmentation_denominator)
            assert nx.algorithms.is_connected(self.construct_graph(self.A)), 'Graph must be connected!'

        self.loss_fn = loss_fn
        self.sampling_fn = sampling_fn
        self.criterion = criterion

        self.scores_matrix_needs_update = True
        self.S = None

        self.reset_weights()

    def construct_graph(self, A):
        return nx.DiGraph(A) if self.directed else nx.Graph(A)

    def reset_weights(self):
        """
        Call when training on a new graph
        """
        self.W_d = (self.gamma * torch.randn(self.N, self.H, device=DEVICE, dtype=DTYPE)).clone().detach().requires_grad_(True)
        self.W_u = (self.gamma * torch.randn(self.H, self.N, device=DEVICE, dtype=DTYPE)).clone().detach().requires_grad_(True)

    def train(self, steps:int=200, lr:float=0.1, weight_decay:float=1e-7, verbose:bool=True):
        """
        Perform training on self.A
        """

        optimizer = torch.optim.Adam(params=[self.W_d, self.W_u], lr=lr, weight_decay=weight_decay)
        A = (torch.tensor(self.A.toarray())).clone().detach()

        for step in range(steps):
            optimizer.zero_grad()
            W = self._get_W()
            loss = self.loss_fn(W, A, self.n_edges)
            loss.backward()
            optimizer.step()

            self.scores_matrix_needs_update = True
            if verbose: print("step", step, "/", steps, ', loss:', loss.item())
            stopping_criterion_is_met = self.criterion is not None and self.criterion(step, self)

            if stopping_criterion_is_met:
                if verbose: print("early stop.")
                break

    
    def sample_graph(self, n_edges=None, self_loops=False):
        """
        Samples graph using self.sampling_fn.
        if n_edges is none, the amount of edges will be controlled by the SamplingFunction
        self_loops controls if the sampled graph is allowed to contain self loops
        """
        return self.sampling_fn(self, n_edges, self_loops)

    def _get_W(self):
        """Returns the learned low-rank approximation"""
        W = self.W_d @ self.W_u
        W = W - W.max(dim=-1, keepdims=True)[0]
        return W

    def _get_synthetic_P(self, W):
        """Rturns the learned transition matrix"""
        P = torch.nn.functional.softmax(W, dim=1).detach().numpy()
        if self.directed:
            # ensure aperiodicity, not used if undirected since it is not used in CELL
            P += np.eye(self.N)/self.N 
        return P

    def _get_S(self, pi, P):
        """Returns the score matrix"""
        S = np.maximum(pi * P, 0)
        return S

    def _get_pi(self, P):
        """Returns the stationary probabilities of P"""
        _, vec = eigs(P.T, k=1, sigma=0.99999, which='LM')
        pi = np.real(vec)
        pi = pi / pi.sum()
        pi = pi.flatten()
        return pi

    def get_scores_matrix(self):
        """Get the scores matrix, only update it if necessary"""
        if self.scores_matrix_needs_update:
            W = self._get_W()
            P = self._get_synthetic_P(W)
            pi = self._get_pi(P)
            self.S = self._get_S(pi, P)
            self.scores_matrix_needs_update = False
        
        return self.S