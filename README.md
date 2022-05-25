# Directed_CELL

This repository contains implementations of generative models for directed and weighted graphs using low-rank approximations. The models are based on CELL, short for Cross-Entropy Low-rank Logits, introduced by Rendsburg et al.[[1]](#1) 

The algorithm works by learning a low-rank approximation of the transition matrix of an input graph, and samples new graphs by sampling edges without replacement from the stationary version of this low-rank transition matrix.

The CELL algorithm, is as-is only compatible with undirected graphs, but by making slight modifications to the learning process and graph sampling method of the algorithm, it is possible to extend it to directed graphs. Two approaches to this are available here, as well as an approach to generate weighted graphs. If you are interested in details, please see ...

### DirectedCELL

The simplest extension, which simply lifts the constraint of the CELL sampling method to produce a symmetric adjacency matrix. A demo of how to use this model can be found in `demo_directed.ipynb`. The model takes several hyperparameters and options:

```{python3}
H # The maximum rank of the learned transition matrix
augmentation_denominator # Needed if the input graph is not strongly connected
lr # Learning rate
weight_decay # L2 regularization
directed # Determines if the model samples directed or undirected graphs

loss_fn # the loss function used during training
sampling_fn # the sampling function used
criterion # early stopping criterion used during training
```


### DirectionClassificationCELL

Instead of modifying CELL directly, this algorithm uses any Graph Generative Model that can generate undirected graphs, and trains a classification model that classifies the edges' "directions" based on an input directed graph. The model is trained on pairs of node embeddings of the input graph. A demo can be found in `demo_direction_classification.ipynb`

### Weighted graph generation

Similar to DirectionClassificationCELL, weights can be added to a graph through training a regression model to predict edge weights based on node embeddings. This is demonstrated in `demo_weighted.ipynb`

## Which model should I use

We find that the following models are the best-performing in terms of reproducing global statistics of the input graph (for directed and unweighted graphs):

* DirectionClassificationCELL with RolX embeddings (see [[2]](#2)). While this approach is the best performing one, it is less scalable than other methods since it is costly to generate node embeddings, especially for very large graphs
* A model that is very close in terms of reproducing global statistics is the DirectedCELL model, using the loss function "LazyLossFunction" and the graph sampling function "SampleGraphLazy" (see `directed_cell/options.py`). If you just want a model that is fast and works reasonably well, this is a good option.

When it comes to weighted graphs, the approach demonstrated in `demo_weighted.ipynb` is the one we have found works best. It can be paired with any approach to generate directed graphs.

## Installation

The implemented models do not require any GPU for training. All requirements can be installed through 

```
pip install -r requirements.txt
```

<a id="1">[1]</a> 
Rendsburg, L., Heidrich, H. &amp; Luxburg, U.V.. (2020). NetGAN without GAN: From Random Walks to Low-Rank Approximations. <i>Proceedings of the 37th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 119:8073-8082 Available from https://proceedings.mlr.press/v119/rendsburg20a.html.

<a id="2">[2]</a> 
Keith Henderson, Brian Gallagher, Tina Eliassi-Rad, Hanghang Tong, Sugato Basu, Leman Akoglu, Danai Koutra, Christos Faloutsos, and Lei Li. 2012. RolX: structural role extraction & mining in large graphs. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '12). Association for Computing Machinery, New York, NY, USA, 1231–1239. https://doi.org/10.1145/2339530.2339723

