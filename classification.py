from __future__ import division
from __future__ import print_function

import os
import glob
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from numpy.lib.function_base import select
from sklearn.model_selection import train_test_split

from cls_utils import load_data, process_data, roc_score
from cls_model import omicsGAT


def train(model, optimizer, features, adj, idx_train, idx_val, labels, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = roc_score(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = roc_score(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.data),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data



def compute_test(model, features, adj, idx_test, labels):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = roc_score(output[idx_test], labels[idx_test])
    
    return acc_test, loss_test.data




def train_single(X, Y, adj, idx_train, idx_val, idx_test, args):
    
    adj, features, labels = process_data(X, Y, adj) #neighbors=args.neighbors)

   
    model = omicsGAT(nfeat=features.shape[1], 
                nhid=args.embed, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    if args.cuda:
        print('cuda selected')
        gpu = torch.device('cuda:0')
        model.cuda(gpu)
        features = features.cuda(gpu)
        adj = adj.cuda(gpu)
        labels = labels.cuda(gpu)
        idx_train = idx_train.cuda(gpu)
        idx_val = idx_val.cuda(gpu)
        idx_test = idx_test.cuda(gpu)

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)   ## for autograd features of the tensors
    

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(model, optimizer, features, adj, idx_train, idx_val, labels, epoch))

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
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
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    acc_test, loss_test = compute_test(model, features, adj, idx_test, labels)

    return acc_test, loss_test


def classifier(args):
    # Load data
    select = args.selection
    X, Y, adj = load_data()
    Y = Y[select]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    x, x_val, y, y_val = train_test_split(X_train, Y_train, test_size = 0.2)

    idx_train = torch.LongTensor(x.index)
    idx_val = torch.LongTensor(x_val.index)
    idx_test = torch.LongTensor(X_test.index)

    X.drop('Samples', axis = 1, inplace=True)
    score, loss = train_single(X, Y, adj, idx_train, idx_val, idx_test, args)

    print("Test set results:",
        "Loss = {:.4f}".format(loss),
        "AUROC score = {:.4f}".format(score))