from abc import abstractmethod, ABC
import networkx as nx
from graphrole import RecursiveFeatureExtractor, RoleExtractor
import node2vec as n2v
import pandas as pd
from hermitian_laplacian import get_embeddings as hermlap, HERMLAP_S, HERMLAP_T
from typing import List
from directed_cell.utils import is_weighted, get_graph

class Embedder(ABC):
    @abstractmethod
    def get_embeddings(self, G):
        """Returns a N*F DataFrame, where N is the number of nodes and F is the number of features"""
        pass

    def embed_multigraph(self, Gs:List[nx.Graph]):
        Ns = [len(G.nodes) for G in Gs]
        
        G = nx.disjoint_union_all(Gs)
        embeddings = self.get_embeddings(G)
        res = []
        idx = 0
        for i in range(len(Gs)):
            res.append(embeddings.iloc[idx:idx+Ns[i]])

        return res

class RolXEmbedder(Embedder):
    """Implementation using https://github.com/dkaslovsky/GraphRole"""
    def __init__(self):
        self.refex = None
        self.rolx = None

    def get_embeddings(self, G, max_generations=10):
        self.refex = RecursiveFeatureExtractor(G, max_generations=max_generations)
        features = self.refex.extract_features()

        self.rolx = RoleExtractor(n_roles=None)
        self.rolx.extract_role_factors(features)

        # self.rolx.roles
        return self.rolx.role_percentage

class Node2VecEmbedder(Embedder):
    """Implementation using https://github.com/eliorc/node2vec"""
    def __init__(self, dimensions:int=128, walk_length:int=80, num_walks:int=10, p:float=1, q:float=1, workers:int=1):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.model = None

    def get_embeddings(self, G):
        model = n2v.Node2Vec(G, self.dimensions, self.walk_length, self.num_walks, self.p, self.q, workers=self.workers)
        self.model = model.fit(window=10, min_count=1, batch_words=4)
        
        embeddings = []
        for N in G.nodes:
            embeddings.append(self.model.wv[N])

        return pd.DataFrame(embeddings)


class HermitianLaplacianEmbedder(Embedder):
    def __init__(self, S=HERMLAP_S, T=HERMLAP_T, q=0.02, verbose=False, normalize=False):
        self.S = S
        self.T = T
        self.q = q
        self.normalize = normalize
        self.verbose = verbose

    def get_embeddings(self, G):
        return pd.DataFrame(hermlap(G, self.S, self.T, self.q, progress=self.verbose, normalize=self.normalize))
    
    def embed_multigraph(self, Gs:List[nx.Graph]):
        if self.S is not None:
            return [self.get_embeddings(G) for G in Gs]
        else:
            return super().embed_multigraph(Gs)
        

####################################################################################################

if __name__ == '__main__':
    G1 = nx.karate_club_graph()
    G2 = nx.karate_club_graph()
    res = HermitianLaplacianEmbedder().embed_multigraph([G1, G2])
    print(res)