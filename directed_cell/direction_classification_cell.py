import numpy as np
import networkx as nx
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
from typing import Union

from abc import ABC, abstractmethod
from directed_cell.embeddings import Embedder

def get_features_from_embeddings(embeddings:pd.DataFrame, graph:nx.Graph):
    """
    Creates features by pairing embeddings for nodes which have an edge between them
    Assumes that there are no bidirectional edges in the graph
    """
    labels = isinstance(graph, nx.DiGraph) # Extract labels as well
    X = []
    y = [] if labels else None
    explored_double_edges = set()
    for U, V in graph.edges:
        if (V, U) in explored_double_edges:
            continue
        X.append(embeddings.iloc[U].values.tolist() + embeddings.iloc[V].values.tolist())
        if labels:

            if graph.has_edge(U, V) and graph.has_edge(V, U):
                y.append(2)
                y.append(2)
                explored_double_edges.add((U, V))
            else:
                y.append(1) # edge goes from first node to last node
                y.append(0)
            X.append(embeddings.iloc[V].values.tolist() + embeddings.iloc[U].values.tolist())
            
    X = np.array(X)
    if y is not None: y = np.array(y)
    return X, y

class EdgeClassifier(ABC):
    """Just here for typing"""
    def __init__(self):
        self.needs_features = True
    @abstractmethod
    def fit(X, y):
        pass

    @abstractmethod
    def predict(X):
        pass

class RandomEdgeClassifier(EdgeClassifier):

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X, **kwargs):
        return np.random.choice([0, 1, 2], size=X.shape[0])



class DirectionClassifier:
    def __init__(self, G: nx.DiGraph, embedder: Embedder, classifier: Union[sklearn.base.ClassifierMixin, EdgeClassifier]):
        """Can use any scikit-learn classifier, or a classifier that uses the same methods"""
        # Create preprocessed version of graph using best augmentation method
        assert G.is_directed()
        self.num_edges = len(G.edges)
        self.G = G
        self.G_undirected = G.to_undirected()
        self.embedder = embedder
        self.classifier = classifier
        self.pipeline = Pipeline(
            steps=[('scaler', MinMaxScaler()), ('classifier', classifier)]
            )

    def train_classifier_and_sample_graph(self, G_hat:nx.Graph, verbose=True):
        assert not G_hat.is_directed()

        # Extract embeddings using the embedder
        # This is done simultaneously on both graphs to support embedders as RolX
        #  where the dimensions and order of dimensions are dependent on the graph
        if verbose: print("Embedding input and output graph")
        E_hat, E = self.embedder.embed_multigraph([G_hat, self.G])

        # Get training data and get inference set
        if verbose: print("Creating features from embeddings")
        X_train, y_train = get_features_from_embeddings(E, self.G)
        X_test, _ = get_features_from_embeddings(E_hat, G_hat)

        # Train classifier
        if verbose: print("Training classifier")
        self.pipeline.fit(X=X_train, y=y_train)
        if verbose: print(classification_report(y_train, self.pipeline.predict(X_train)))
    
        # Predict directions of edges in synthetic graph
        preds = self.pipeline.predict(X_test)
        A_hat = sp.lil_matrix(nx.to_scipy_sparse_matrix(G_hat))
        for (U, V), label in zip(G_hat.edges, preds):
            if label == 2:
                continue
            to_remove = (V, U) if label == 1 else (U, V)
            A_hat[to_remove] = 0

        A_hat = sp.csr_matrix(A_hat)
        A_hat.eliminate_zeros()
        print(A_hat.shape)
        return A_hat