import numpy as np
from scipy.stats import norm
import pandas as pd
from typing import List
from joblib import Parallel, delayed
import networkx as nx

def confidence_interval(X:np.ndarray, alpha:float=0.05):
    mu, sigma = np.mean(X), np.std(X, ddof=1)
    standard_err = norm.ppf(1-alpha/2)*sigma/np.sqrt(X.shape[0])
    CI = [mu - standard_err, mu + standard_err]
    return CI, mu, standard_err


class EvaluationPipeline:
    """Helper class to more easily evaluate the performance of a GGM"""
    def __init__(self, metric_functions: List[callable], metric_names: List[str], pairwise_metric_functions: List[callable]=[], pairwise_metric_names: List[str]=[], verbose=True):
        assert len(metric_functions) == len(metric_names), f"Supplied metric names must be equal to the amount of metrics ({len(metric_names)} vs {len(metric_functions)})"
        self.metric_functions = metric_functions
        self.metric_names = metric_names
        assert (len(pairwise_metric_functions)==0 and len(pairwise_metric_names)==0) or len(pairwise_metric_functions) == len(pairwise_metric_names), "length must be equal"
        self.pairwise_metric_functions = pairwise_metric_functions
        self.pairwise_metric_names = pairwise_metric_names
        self.G_metrics = None
        self.verbose = verbose

    def evaluate_G(self, G:nx.DiGraph) -> pd.DataFrame:
        """Evaluate a single graph"""
        G_metrics = []
        for i, metric_func in enumerate(self.metric_functions):
            if self.verbose: print("evaluating", self.metric_names[i])
            G_metrics.append(metric_func(G))
        
        stats = pd.DataFrame()
        stats['metric'] = self.metric_names
        stats['value'] = G_metrics

        return stats

    def evaluate_G_hats(self, G_hats, alpha=0.05, try_parallelize=False, n_jobs=4) -> pd.DataFrame:
        """Evaluate a set of graphs"""
        n_funcs = len(self.metric_functions)
        G_hat_metrics = np.zeros((n_funcs, len(G_hats)))
        CI_lower = []
        CI_upper = []
        std_err = []
        mean = []

        for i, metric_func in enumerate(self.metric_functions):
            if self.verbose: print("evaluating", self.metric_names[i])
            
            if try_parallelize and len(G_hats) > 1:
                metrics = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(metric_func)(G_hat) for G_hat in G_hats)
            else:
                metrics = []
                for i in range(len(G_hats)):
                    metrics.append(metric_func(G_hats[i]))
           
            G_hat_metrics[i, :] = metrics
            
            if len(G_hats) > 1:
                CI, mu, standard_err = confidence_interval(G_hat_metrics[i, :], alpha=alpha)
            else:
                CI, mu, standard_err = [None, None], G_hat_metrics[i, :][0], None

            CI_lower.append(CI[0])
            CI_upper.append(CI[1])
            mean.append(mu)
            std_err.append(standard_err)

        stats = pd.DataFrame()
        stats['metric'] = self.metric_names
        stats['ci_'+str(1-alpha)+'l'] = CI_lower
        stats['ci_'+str(1-alpha)+'u'] = CI_upper
        stats['synth. mean'] = mean
        stats['synth. std. err.'] = std_err

        return stats

    def evaluate_pairwise(self, G, G_hats, alpha=0.05) -> pd.DataFrame:
        """Evaluate a set of graphs against an input graph using pairwise metrics (distances, MMD etc.)"""
        n_funcs = len(self.pairwise_metric_functions)
        pairwise_metric_values = np.zeros((n_funcs, len(G_hats)))
        CI_lower = []
        CI_upper = []
        std_err = []
        mean = []

        for i, pairwise_metric_func in enumerate(self.pairwise_metric_functions):
            if self.verbose: print("evaluating", self.pairwise_metric_names[i])
            for o, G_hat in enumerate(G_hats):
                print(f"graph {o}")
                pairwise_metric_values[i, o] = pairwise_metric_func(G, G_hat)

            if len(G_hats) > 1:
                CI, mu, standard_err = confidence_interval(pairwise_metric_values[i, :], alpha=alpha)
            else:
                CI, mu, standard_err = [None, None], pairwise_metric_values[i, :][0], None

            CI_lower.append(CI[0])
            CI_upper.append(CI[1])
            mean.append(mu)
            std_err.append(standard_err)

        stats = pd.DataFrame()
        stats['metric'] = self.pairwise_metric_names
        stats['ci_'+str(1-alpha)+'l'] = CI_lower
        stats['ci_'+str(1-alpha)+'u'] = CI_upper
        stats['synth. mean'] = mean
        stats['synth. std. err.'] = std_err

        return stats
        
def directed_evaluation_pipeline(verbose=True) -> EvaluationPipeline:
    import directed_cell.metrics as met

    eval_pipe = EvaluationPipeline(
        metric_functions=[
            lambda A : met.in_degree(A, 'max'),
            lambda A : met.in_degree(A, 'min'),
            lambda A : met.out_degree(A, 'max'),
            lambda A : met.out_degree(A, 'min'),
            lambda A : met.power_law_exponent(A, 'in'),
            lambda A : met.power_law_exponent(A, 'out'),
            lambda A : met.gini_coef(A, 'in'),
            lambda A : met.gini_coef(A, 'out'),
            met.assortativity,
            met.average_local_clustering_coefficient,
            met.undirected_wedge_count,
            met.directed_wedge_count,
            met.undirected_triangle_count,
            met.directed_triangle_count,
            met.undirected_square_count,
            met.directed_square_count,
            met.characteristic_path_length,
            met.diameter,
            met.largest_scc
        ],
        metric_names=[
            'max. in-deg', 
            'min. in-deg', 
            'max. out-deg', 
            'min. out-deg', 
            'power law exp. (in)', 
            'power law exp. (out)', 
            'gini coef. (in)', 
            'gini coef (out)', 
            'assortativity', 
            'avg. loc. clust. coef.', 
            'und. wedge count',
            'dir. wedge count',
            'und. triangle count',
            'dir. triangle count',
            'und. square count',
            'dir. square count',
            'char. path. len.',
            'diameter',
            'largest scc'],
        verbose=verbose
    )

    return eval_pipe


def weighted_evaluation_pipeline():
    import directed_cell.metrics as met
    eval_pipe = EvaluationPipeline(
        metric_functions=[
            lambda A : met.in_degree(A, 'max'),
            lambda A : met.in_degree(A, 'min'),
            lambda A : met.out_degree(A, 'max'),
            lambda A : met.out_degree(A, 'min'),
            lambda A : met.out_degree(A, 'mean'),
            lambda A : met.assortativity(A, weight="weight"),
            met.weighted_clustering_coefficient,
            lambda A : met.power_law_exponent(A, 'in'),
            lambda A : met.power_law_exponent(A, 'out'),
            lambda A : met.gini_coef(A, 'in'),
            lambda A : met.gini_coef(A, 'out'),
        ],
        metric_names=[
            'weighted max. in-deg', 
            'weighted min. in-deg', 
            'weighted max. out-deg', 
            'weighted min. out-deg', 
            'weighted avg. deg', 
            'weighted assortativity', 
            'weighted clustering coef.', 
            'weighted power law exp. (in)', 
            'weighted power law exp. (out)', 
            'weighted gini coef. (in)', 
            'weighted gini coef (out)', 
            ]
    )
    
    return eval_pipe