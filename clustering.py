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