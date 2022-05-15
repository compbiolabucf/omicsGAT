import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer



class omicsGAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout=0, alpha=0.2):
        super(omicsGAT, self).__init__()
        self.dropout = dropout

        ## creating attention layers for given number of heads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] 
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)     ## adding the modules for each head

        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  ## concatanating all the attention heads dimension (#input X out_features*nb_heads);   out_features = nhid... each head contributing to (#input X out_features)

        return x   







class autoencoder(nn.Module):
    def __init__(self,in_features, nhid, nheads, alpha=0.2):
        super(autoencoder, self).__init__()
        self.encoder = omicsGAT(in_features, nhid, nheads, alpha=alpha)

        embedding = nhid*nheads
        self.decoder = nn.Sequential(
                        nn.Linear(embedding, int(embedding/2)),
                        nn.ReLU(),
                        nn.Linear(int(embedding/2), int(in_features/4)),
                   
                        nn.ReLU(),
                        nn.Linear(int(in_features/4), int(in_features/2)),
                   
                        nn.ReLU(),
                        nn.Linear(int(in_features/2), in_features))
        
        
    def forward(self, x, adj, train=True):
        out = self.encoder(x, adj)
        
        if train:
            out = self.decoder(out)
        
        return out

