import numpy as np
import torch
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score




def roc_score(output, labels):
    preds = output.max(1)[1]
    out_arr = preds.cpu().detach().numpy()
    labels_arr = labels.cpu().detach().numpy()
    score = roc_auc_score(labels_arr, out_arr)
    return score



def load_data():
    
    X_load = pd.read_csv('brca_data/feature_matrix.csv')
    Y_load = pd.read_csv('brca_data/labels.csv')
    adj = pd.read_csv('brca_data/adjacency_matrix.csv')
    
    Y_load.drop('Samples', axis = 1, inplace=True)
    Y_load.replace(to_replace={'Positive', 'Negative'}, value ={1,0}, inplace = True)
    adj.drop('Unnamed: 0', axis=1, inplace=True)

    return X_load, Y_load, adj



def process_data(X, Y, adj):
    
    features = csr_matrix(X)
    labels = np.array(Y)
    adj = csr_matrix(adj)
    
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    return adj, features, labels


