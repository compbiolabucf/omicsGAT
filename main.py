from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
from classification import classifier
from clustering import stratifier


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='classification', help='Selects the type of task to perform.')
parser.add_argument('--clustering_type', type=str, default='bulk', help='If clustering is to be performed, selects the type of clustering (bulk or single-cell) to perform.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=27, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--selection', type=str, default='ER', help='Selects the category of BRCA: ER or TN')
parser.add_argument('--neighbors', type=int, default=10, help='No. of neighbors to keep for a node')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--embed', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument('--nb_clusters', type=int, default=5, help='Number of clusters')
parser.add_argument('--cluster_affn', type=str, default='manhattan', help='Clustering affinity used for hierarchical clustering')
parser.add_argument('--cluster_dist', type=str, default='average', help='Clustering distance used for hierarchical clustering')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## setting random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if (args.task == 'classification'):
    classifier(args)
elif (args.task == 'clustering'):
    stratifier(args)