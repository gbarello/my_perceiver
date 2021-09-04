import torch
from torch import nn
import numpy as np

def get_positional_embedding(data_size, n_freq, min_lam = 1):
    coordinates = np.meshgrid(*[np.linspace(-1,1,d) for d in data_size])
    scales = [np.logspace(np.log10(1),np.log10(int((d-1)/min_lam/2)),n_freq) for d in data_size]
    
    features = np.concatenate([[[np.cos(c*np.pi*s),np.cos(c*np.pi*s)] for s in ss] for c,ss in zip(coordinates, scales)],0)
    return np.concatenate([features[:,0],features[:,1],np.transpose(np.stack(coordinates,-1),[2,0,1])],0)
    
def prep_data(data,pos_enc,add_enc = True):
    if add_enc:
        pos_reshape = torch.unsqueeze(pos_enc,0).repeat([data.shape[0],1,1,1])
        data = torch.cat([data,pos_reshape],1)
    
    data = torch.transpose(data,1,3)
    data = torch.flatten(data,1,2)
    return data
