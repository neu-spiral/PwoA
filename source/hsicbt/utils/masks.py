from scipy.fftpack import fft
import scipy.io as spio
import pickle
import numpy as np
import torch
import yaml
import copy
import os

def load_layer_config(args,model,task):
    # fixed_layer: shared layers for all tasks while config is set to be 0
    
    
    prune_ratios = {}
    pruned_layer = []

    # For fixed layer
    fixed_layer = []
    if args.dataset == 'cifar' or args.dataset == 'mixture':
        fixed_layer = ['module.fc1.bias']
    elif args.dataset == 'mnist':
        fixed_layer = ['module.fc1.bias','module.fc2.bias']

    # For output layer
    output_layer = []
    if args.dataset == 'cifar' or args.dataset == 'mixture':
        output_layer = ['module.fc2.weight','module.fc2.bias']
    elif args.dataset == 'mnist':
        output_layer = ['module.fc3.weight','module.fc3.bias']
    elif args.dataset == 'rfmls':
        output_layer = ['module.linear.weight','module.linear.bias']
    # For pruned layer
    config_setting = list(map(float, args.config_setting.split(",")))


    if len(config_setting) == 1:
        sparse_setting = float(config_setting[0])
    else:
        sparse_setting = float(config_setting[task])/float(sum(config_setting))

    sparse_setting = 1-sparse_setting*args.config_shrink

    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if args.dataset == 'cifar' or args.dataset == 'mixture':
                if 'weight' in name and name!='module.fc2.weight' and name not in fixed_layer:
                #if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)
            elif args.dataset == 'mnist':
                if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                #if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)
            elif args.dataset == 'rfmls':
                if 'weight' in name and name!='module.linear.weight' and name!='module.linear.bias':
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)

    args.prune_ratios = prune_ratios
    print('Pruned ratio:',sparse_setting)
            
    args.pruned_layer = pruned_layer
    args.fixed_layer = fixed_layer
    args.output_layer = output_layer
    print('Pruned layer:',pruned_layer)
    print('Fixed layer:',fixed_layer)
    print('Output layer:',output_layer)
    return args

def model_loader(args):
    if args.adaptive_mask:
        from models.masknet import CifarNet, MnistNet
    else:
        from models.cifarnet import CifarNet
        from models.mnistnet import MnistNet
    
    if args.arch == 'cifarnet':
        model = CifarNet(args.input_size, args.classes)
    elif args.arch == 'mnistnet':
        model = MnistNet(args.input_size, args.classes)
 
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    
    return model

def mask_joint(args,mask1,mask2):
    '''
    mask1 has more 1 than mask2
    return: new mask with only 0s and 1s
    '''

    masks = copy.deepcopy(mask1)
    if not mask2:
        return mask1
    for name in mask1:
        if name not in args.fixed_layer and name in args.pruned_layer:
            non_zeros1,non_zeros2 = mask1[name], mask2[name]
            non_zeros = non_zeros1 + non_zeros2
            
            # Fake float version of |
            under_threshold = non_zeros < 0.5
            above_threshold = non_zeros > 0.5
            non_zeros[above_threshold] = 1
            non_zeros[under_threshold] = 0
            
            masks[name] = non_zeros
    return masks

def mask_reverse(args, mask):
    mask_reverse = copy.deepcopy(mask)
    for name in mask:
        if name in args.pruned_layer:
            mask_reverse[name] = 1.0-mask[name]
    return mask_reverse

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

def cumulate_model(args, task):
    '''
    Cumulate models for individual task.
    '''
    state_dict = {}
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    
    #state_dict = torch.load(save_path + "/retrained.pt")
    # Trigger for experiment [leave space for future learning]
    if task < args.tasks-1:
        state_dict = torch.load(save_path + "/retrained.pt")
    else: # for last task
        state_dict = torch.load(save_path +"/{}{}.pt".format(args.arch, args.depth) )
            
    if 0 < task:
        save_path = os.path.join(args.save_path_exp,'task'+str(task-1))
        state_dict_prev = torch.load(save_path + "/cumu_model.pt")
        for name, param in state_dict_prev.items():
            if name in args.pruned_layer:
                state_dict[name].copy_(state_dict[name].data + param.data)
    
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    torch.save(state_dict, save_path+"/cumu_model.pt")

def set_adaptive_mask(model, reset=False, assign_value='', requires_grad=False):
    for name, W in (model.named_parameters()):
        if 'mask' in name:
            
            # set mask to be one
            if reset:
                weight = W.cpu().detach().numpy()
                W.data = torch.ones(weight.shape).cuda()
            
            # set mask to be given value
            elif assign_value:
                weight_name = name.replace('w_mask', 'weight')
                if weight_name in assign_value:
                    W.data = assign_value[weight_name].cuda()
                
            W.requires_grad = requires_grad
            
def load_state_dict(args, model, state_dict, target_keys=[], masks=[]):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. The keys of :attr:`state_dict` must
    exactly match the keys returned by this module's :func:`state_dict()`
    function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        masks: set target params to be 1.
    """
    own_state = model.state_dict()
    
    if target_keys:
        for name, param in state_dict.items():
            if name in target_keys:       # changed here
                if name not in own_state:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                param = param.data
                own_state[name].copy_(param)
    elif masks:
        print('Loading layer...')
        for name, param in state_dict.items():
            if name in args.pruned_layer:     # changed here
                param = param.data
                param_t = own_state[name].data
                mask = masks[name].cuda()
                own_state[name].copy_(param + param_t*mask)
                #print(name)
    else:
        print('Loading layer...')
        for name, param in state_dict.items():
            if name not in own_state:     # changed here
                continue
            param = param.data
            own_state[name].copy_(param)
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.3 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr