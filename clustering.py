from __future__ import division
from __future__ import print_function

import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sklearn.metrics as sm
from sklearn.cluster import AgglomerativeClustering
import scipy.spatial as sp

from cluster_model import autoencoder
from cluster_utils import load_data, process_data


# # Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
# parser.add_argument('--seed', type=int, default=27, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
# parser.add_argument('--nb_clusters', type=int, default=5, help='Number of clusters')
# # parser.add_argument('--net_density', type=int, default=0.2, help='Network density of the adjacency matrix')
# parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--embed', type=int, default=64, help='Number of hidden units.')
# parser.add_argument('--nb_heads', type=int, default=64, help='Number of head attentions.')
# # parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--patience', type=int, default=100, help='Patience')
# parser.add_argument('--cluster_affn', type=str, default='manhattan', help='Clustering affinity used for hierarchical clustering')
# parser.add_argument('--cluster_dist', type=str, default='average', help='Clustering distance used for hierarchical clustering')




## suitable seed-- 27, 8, 1, 7, 2, 16
def train_auto(auto_model, optimizer, loss, features, adj, epoch):
    t = time.time()
    auto_model.train()
    optimizer.zero_grad()
    output = auto_model(features, adj, train=True)
    
    
    loss_train = loss(features, output)
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data))

    return loss_train.data


    
def stratifier(args):

    X, Y, adj = load_data(args.clustering_type)

    adj, features, labels = process_data(X, Y, adj)
    auto_model = autoencoder(features.shape[1], 
                nhid=args.embed, 
                nheads=args.nb_heads,  
                alpha=args.alpha)
    auto_optimizer = optim.Adam(auto_model.parameters(), 
                    lr=args.lr, 
                    weight_decay=args.weight_decay)

    auto_loss = nn.MSELoss()

    if args.cuda:
        print('cuda selected')
        gpu = torch.device('cuda:0')
        auto_model.cuda(gpu)
        features = features.cuda(gpu)
        adj = adj.cuda(gpu)
        labels = labels.cuda(gpu)

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)   ## for autograd features of the tensors




    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best_epoch = 0


    for epoch in range(args.epochs):
        loss_values.append(train_auto(auto_model, auto_optimizer, auto_loss, 
                                        features, adj, epoch))

        if epoch==0:
            best = loss_values[-1]

        torch.save(auto_model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    auto_model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))





    ####### Hierarchical Clustering ########

    auto_out = auto_model(features, adj, train=False)

    f = auto_out.cpu().detach().numpy()
    l = labels.cpu().detach().numpy()


    ## clustering
    clustering = AgglomerativeClustering(n_clusters = args.nb_clusters, affinity=args.cluster_affn, linkage = args.cluster_dist, compute_distances=True)
    clustering.fit(f)

    NMI = sm.normalized_mutual_info_score(l.squeeze(), clustering.labels_)
    ARI = sm.adjusted_rand_score(l.squeeze(), clustering.labels_)


    print('NMI: '+str(NMI)+' ARI: '+str(ARI))