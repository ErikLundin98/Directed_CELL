import scipy
import numpy as np
import networkx as nx

# doi:10.1137/0205044
# see https://en.wikipedia.org/wiki/Strong_connectivity_augmentation
# implementation based on the correction made by https://terpconnect.umd.edu/~raghavan/preprints/stc.pdf

def augment(G:nx.DiGraph, E:list, copy=True, **attr):
    if copy: G = G.copy()
    for U, V in E:
        G.add_edge(U, V, **attr)
    return G

def get_condensation(G):
    scc = list(nx.algorithms.strongly_connected_components(G))
    D_scc = nx.algorithms.components.condensation(G, scc)
    nodes = D_scc.nodes
    sources = [node for node in nodes if D_scc.in_degree(node) == 0 and D_scc.out_degree(node) > 0]
    sinks = [node for node in nodes if D_scc.in_degree(node) > 0 and D_scc.out_degree(node) == 0]
    X = [node for node in nodes if D_scc.degree(node) == 0] # isolated_nodes

    return scc, D_scc, sources, sinks, X

def STRONGCONNECT(A):
    """
    Unweighted implementation of Tarjan's edge augmentation algorithm
    Returns the minimal set of edges needed to make the graph A strongly connected
    """
    if isinstance(A, np.ndarray):
        D = nx.DiGraph(A)
    elif isinstance(A, nx.DiGraph):
        D = A.copy()
    else:
        exit("Not a numpy array or np array")
    
    # SC1: Form the condensation of D
    scc, D_scc, sources, sinks, X = get_condensation(D)
    flipped = False
    if len(sources) > len(sinks):
        flipped = True
        D = D.reverse()
        scc, D_scc, sources, sinks, X = get_condensation(D)

    
    # SC2: Apply STCORRECT to find set of sources and sinks
    V, W, p = STCORRECT(D_scc, sources) # V sources, W sinks
    # The lists V, W are "ordered sets" containing the first source and sinks that should be connected
    # But the sets need to be appended with the remaining sources and sinks
    rest_of_sources = list(set(sources) - set(V))
    rest_of_sinks = list(set(sinks) - set(W))

    V = V + rest_of_sources
    W = W + rest_of_sinks

    # SC3 Construct augmenting set of arcs A_asc
    s = len(sources) # n sources
    t = len(sinks) # n sinks
    q = len(X) # n isolated nodes


    a1 = {(W[i-1], V[i]) for i in range(1, p)}
    a2 = {(W[i-1], V[i-1]) for i in range(p+1, s+1)}
    s1 = a1 | a2
    s2 = None
    
    if q == 0 and s == t:
        s2 = {(W[p-1], V[0])}
    elif q == 0 and s < t:
        t1 = {(W[p-1], W[s])}
        t2 = {(W[i-1], W[i]) for i in range(s+1, t)}
        t3 = {(W[t-1], V[0])}
        s2 = t1 | t2 | t3
    else:
        t1 = {(W[p-1], W[s])}
        t2 = {(W[i-1], W[i]) for i in range(s+1, t)}
        t3 = {(W[t-1], X[0])}
        t4 = {(X[i-1], X[i]) for i in range(1, q)}
        t5 = {(X[q-1], V[0])}
        s2 = t1 | t2 | t3 | t4 | t5

    A_asc = s1 | s2 # edge set to add to the condensed graph
    
    # SC4 Convert A_asc into augmenting set of edges A_aug for D using Lemma 1 in paper
    to_add = []
    for edge_to_add in A_asc:
        if flipped: edge_to_add = (edge_to_add[1], edge_to_add[0])
        # identify corresponding nodes in original graph D
        from_node = next(iter(scc[edge_to_add[0]]))
        to_node = next(iter(scc[edge_to_add[1]]))
        to_add.append((from_node, to_node))

    return to_add
    
    


def STCORRECT(D_scc, sources):
    """Returns an ordering of sources and sinks that are used to find the minimal augmenting set of edges to make a graph SC"""

    marked = {node:False for node in D_scc.nodes}

    V = [] # ordering of sources
    W = [] # ordering of sinks
    i = 0
    while True:
        unmarked_sources = [source for source in sources if not marked[source]]
        if len(unmarked_sources) == 0:
            break
        v = unmarked_sources[0]
        w = -1
        sink_not_found = True

        marked, sink_not_found, w = SEARCH(v, D_scc, marked, sink_not_found, w)
        if w != -1:
            i += 1
            if v not in V: V.append(v)
            if w not in W: W.append(w)
    p = i
    return V, W, p
    
def SEARCH(x:int, G:nx.DiGraph, marked:dict, sink_not_found:bool, w:int):
    if not marked[x]:
        # check if x is a sink
        if G.out_degree(x)==0 and G.in_degree(x)>0:
            w = x
            sink_not_found = False

        marked[x] = True
        
        for edge in G.out_edges(x):
            y = edge[1]
        # for edge in G.edges(x):
        #     y = edge[0] if x != edge[0] else edge[1]
            if sink_not_found:
                marked, sink_not_found, w = SEARCH(y, G, marked, sink_not_found, w)


    return marked, sink_not_found, w