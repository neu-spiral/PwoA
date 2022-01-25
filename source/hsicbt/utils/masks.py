import numpy as np
import torch
import yaml
import copy
import os

def set_model_mask(model,mask):
    '''
    mask:{non-zero:1 ; zero:0}
    '''
    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if name in mask:
                W.data *= mask[name].cuda()

def get_model_mask(model):
    masks = {}
    device = next(model.parameters()).device
    
    for name, W in (model.named_parameters()):
        if 'mask' in name:
            continue
        weight = W.cpu().detach().numpy()
        non_zeros = (weight != 0)
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros)
        W = torch.from_numpy(weight).to(device)
        W.data = W
        masks[name] = zero_mask.to(device)
        #print(name,zero_mask.nonzero().shape)
    return masks
            