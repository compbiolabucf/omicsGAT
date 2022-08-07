import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class omicsGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(omicsGAT, self).__init__()
        self.dropout = dropout

        ## creating attention layers for given number of heads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] 
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)     ## adding the modules for each head
        
        in_features = nhid * nheads
        
        self.dnn = nn.Sequential(
                    nn.BatchNorm1d(in_features),
                    nn.ReLU(inplace = True),
                    nn.Linear(in_features,nclass))
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  ## concatanating all the attention heads dimension (#input X out_features*nb_heads);   out_features = nhid... each head contributing to (#input X out_features)
        x = F.dropout(x, self.dropout, training=self.training)        
        x = self.dnn(x)
        x = F.log_softmax(x, dim = 1)

        return x   
