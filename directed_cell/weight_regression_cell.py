import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score

from directed_cell.embeddings import Embedder
from directed_cell.utils import to_unweighted, is_weighted



def regression_report(y_true, y_pred):
    
    error = y_true - y_pred
    percentil = [5,25,50,75,95]
    percentil_value = np.percentile(error, percentil)
    
    metrics = [
        ('mean absolute error', mean_absolute_error(y_true, y_pred)),
        ('median absolute error', median_absolute_error(y_true, y_pred)),
        ('mean squared error', mean_squared_error(y_true, y_pred)),
        ('max error', max_error(y_true, y_pred)),
        ('r2 score', r2_score(y_true, y_pred)),
        ('explained variance score', explained_variance_score(y_true, y_pred))
    ]
    
    print('Metrics for regression:')
    for metric_name, metric_value in metrics:
        print(f'{metric_name:>25s}: {metric_value: >20.3f}')
        
    print('\nPercentiles:')
    for p, pv in zip(percentil, percentil_value):
        print(f'{p: 25d}: {pv:>20.3f}')


def get_features_from_embeddings(embeddings:pd.DataFrame, graph:nx.Graph, labels=True):
    """
    Creates features by pairing embeddings for nodes which have an edge between them
    Assumes that there are no bidirectional edges in the graph
    """
    X = []
    y = [] if labels else None
    for U, V, a in graph.edges.data():
        X.append(embeddings.iloc[U].values.tolist() + embeddings.iloc[V].values.tolist())
        if labels:
            y.append(a['weight'])

    if y is not None: y = np.array(y).astype(float)
    return pd.DataFrame(X), y

class WeightRegressor():
    
    def __init__(self, G: nx.DiGraph, regression_model:sklearn.base.RegressorMixin, embedder:Embedder):
        # Create preprocessed version of graph using best augmentation method
        assert is_weighted(G) and isinstance(G, nx.DiGraph)
        self.G = G
        self.G_unweighted = to_unweighted(G)
        assert not is_weighted(self.G_unweighted)
        self.num_edges = len(G.edges)
        self.G = G
        self.embedder = embedder
        self.regression_model = regression_model
        self.pipeline = Pipeline(
            steps=[('scaler', MinMaxScaler()), ('regression_model', self.regression_model)]
        )

    def train_regression_model_and_sample_graph(self, G_hat : nx.DiGraph, verbose=True, nonzero=True):
        """Should be called after training the CELL model!"""
        assert not is_weighted(G_hat) and isinstance(G_hat, nx.DiGraph)
        A_hat = nx.to_scipy_sparse_matrix(G_hat, format='lil')
        # Extract embeddings using the embedder
        # This is done simultaneously on both graphs to support embedders as RolX
        #  where the dimensions and order of dimensions are dependent on the graph
        if verbose: print("Embedding input and output graph")
        E_hat, E = self.embedder.embed_multigraph([G_hat, self.G_unweighted]) # embeddings for directed (but unweighted graphs)

        # Get training data and get inference set
        if verbose: print("Creating features from embeddings")
        X_train, y_train = get_features_from_embeddings(E, self.G)
        X_test, _ = get_features_from_embeddings(E_hat, G_hat, labels=False)
        
        # Train regressor
        if verbose: print("Training regressor")
        self.pipeline.fit(X=X_train, y=y_train)
        if verbose: regression_report(y_train, self.pipeline.predict(X_train))

        train_preds = self.pipeline.predict(X_train)
        print("predictions during training:", train_preds.min(), train_preds.max())
        
        # Predict weights of edges in synthetic graph
        preds = self.pipeline.predict(X_test)
        print("predictions on test set", preds.min(), preds.max())
        if nonzero and np.any(preds < 0):
            preds[preds < 0] = 0
            if verbose: print("Negative predictions encountered - clipping to 0")
        A_hat = sp.lil_matrix(A_hat)
        for (U, V), predicted_weight in zip(G_hat.edges, preds):
            A_hat[U, V] = predicted_weight

        A_hat = sp.csr_matrix(A_hat)
        A_hat.eliminate_zeros()
        return A_hat