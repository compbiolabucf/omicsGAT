import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.sparse import csr_matrix



def load_data(clr_type):
    
    if (clr_type == 'bulk'): 
        X_load = pd.read_csv('blca_data/feature_matrix.csv', index_col=0)
        Y_load = pd.read_csv('blca_data/labels.csv', index_col=0)
        adjacent_mat = pd.read_csv('blca_data/adjacency_matrix.csv', index_col=0)
    elif (clr_type == 'single_cell'): 
        X_load = pd.read_csv('scr_data/feature_matrix.csv', index_col=0)
        Y_load = pd.read_csv('scr_data/labels.csv', index_col=0)
        adjacent_mat = pd.read_csv('scr_data/adjacency_matrix.csv', index_col=0)

    return X_load, Y_load, adjacent_mat




def process_data(X, Y, adj): 
    
    features = csr_matrix(X)
    labels = np.array(Y)
    adj = csr_matrix(adj)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    return adj, features, labels

