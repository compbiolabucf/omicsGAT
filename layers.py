import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))   ## declaring the weights for linear transformation
        nn.init.xavier_uniform_(self.W.data, gain=1)                           ## initializing the linear transformation weights from the uniform distribution U(-a,a)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))           ## declaring weights for creating self attention coefficients
        nn.init.xavier_uniform_(self.a.data, gain=1)                           ## initializing the attention-coefficient weights
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)    ## multiplying inputs with the weights for linear transformation with dimension (#input X out_features)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = torch.zeros_like(e)                       
        attention = torch.where(adj > 0, e, zero_vec)             ## assigning values of 'e' to those which has value>0 in adj matrix
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)                      ## multiplying attention co-efficients with the input  -- dimension (#input X out_features)

        if self.concat:
            xtra = F.elu(h_prime)
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, h):
        
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        e = h1 + h2.T   # broadcast add
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
