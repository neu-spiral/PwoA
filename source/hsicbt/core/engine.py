import collections

from .. import *
from .  import *

from .train_misc     import *
from .train_standard import *
from .train_hsic_prune     import *

from ..math.admm import *
from ..utils.masks import *
from ..utils.path    import *
from ..utils.io    import *


import torch
import time

def training_standard(config_dict):
    """
    Train model with HBaR or CE only
    """
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['batch_size'])
    torch.manual_seed(config_dict['seed'])

    model = model_distribution(config_dict)
    
    # load pretrained model
    if 'load_model' in config_dict:
        model = load_state_dict(model, get_model_path("{}".format(config_dict['load_model'])))
    model = model.to(config_dict['device'])
    
    # construct single output model for test
    #config_dict['robustness'] = True
    #model_single_output = model_distribution(config_dict)
    #model_single_output = model_single_output.to(config_dict['device'])
    
    nepoch = config_dict['epochs']
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader, \
                                         config_dict['optimizer'], config_dict['learning_rate'], nepoch)

    best = 0
    log_dict = {}
    batch_log_list = []
    epoch_log_dict = collections.defaultdict(list)
    epoch_time = meter.AverageMeter()
    
    for cepoch in range(0, nepoch+1):
        if cepoch > 0:
            start_time = time.time() 
            if config_dict['training_type'] == 'hsictrain':
                hsic_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            elif config_dict['training_type'] == 'backprop':
                standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            else:
                raise ValueError("Unknown training type or not support [{}]".format(config_dict['training_type']))
            epoch_time.update(time.time()-start_time)
            
        epoch_log_dict, best = eval_and_save(config_dict, model, test_loader, 
                                             epoch_log_dict, cepoch, nepoch, best)
        
    log_dict['epoch_log_dict'] = epoch_log_dict
    log_dict['config_dict'] = config_dict
    filename = "{}.npy".format(os.path.splitext(config_dict['model_file'])[0])
    save_logs(log_dict, get_log_filepath("{}".format(filename)))
    print('Overall training time is {:.2f}s.'.format(epoch_time.sum))
    
    return batch_log_list, epoch_log_dict

def training_hsic_prune(config_dict):
    """
    Train hsic model parameters with hsic + hard pruning + masked retrain
    """
    
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['batch_size'])
    torch.manual_seed(config_dict['seed'])
    
    # load pre-trained model
    model = model_distribution(config_dict)
    model = load_state_dict(model, get_model_path("{}".format(config_dict['load_model'])))
    model = model.to(config_dict['device'])
    
    # construct single output model for test
    #config_dict['robustness'] = True
    #model_single_output = model_distribution(config_dict)
    #model_single_output = model_single_output.to(config_dict['device'])
    
    # distillation teacher model
    if config_dict['distill']:
        pretrained = deepcopy(model)
        #pretrained.load_state_dict(torch.load(config_dict['distill_model_path']))
        pretrained = load_state_dict(pretrained, config_dict['distill_model_path'])
        pretrained = pretrained.to(config_dict['device'])
        pretrained.eval()
        config_dict['pretrained'] = pretrained
        if config_dict['distill_loss'] == 'kl':
            distillation_criterion = torch.nn.KLDivLoss(log_target=True).to(config_dict['device'])
        elif config_dict['distill_loss'] == 'mse' or 'mseml':
            distillation_criterion = torch.nn.MSELoss().to(config_dict['device'])
        config_dict['distillation_criterion'] = distillation_criterion

    # optimizer
    nepoch = config_dict['epochs']
    re_nepoch = config_dict['retrain_ep']
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader, \
                                         config_dict['optimizer'], config_dict['learning_rate'], nepoch)
    best = 0
    log_dict = {}
    batch_log_list = []
    epoch_log_dict = collections.defaultdict(list)
    epoch_time = meter.AverageMeter()
    
    # Initializing ADMM; if not admm, do hard pruning only
    admm = ADMM(config_dict, model, rho=config_dict['rho']) if config_dict['admm'] else None
    
    for cepoch in range(0, nepoch+1):
        if cepoch > 0:
            start_time = time.time() 
            if config_dict['training_type'] == 'hsictrain':
                #Train hsic model parameters with (backprop + hisc) + pruning, in the end of each epoch, do ADMM
                hsic_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict, ADMM=admm)
            elif config_dict['training_type'] == 'backprop':
                #Train hsic model parameters with backprop + pruning, in the end of each epoch, do ADMM
                standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict, ADMM=admm)
            else:
                raise ValueError("Unknown training type or not support[{}]".format(config_dict['training_type']))
            epoch_time.update(time.time()-start_time)
        
        if (cepoch-1) % config_dict['admm_epochs'] == 0:
        #model_single_output.load_state_dict(model.state_dict())
        #model_single_output.eval()
            epoch_log_dict, best = eval_and_save(config_dict, model, test_loader, epoch_log_dict, cepoch, nepoch, best, False)
                
        print(get_lr(optimizer))
        
        
    # If not admm, do hard pruning only
    filename = os.path.splitext(config_dict['model_file'])[0]
    save_model(model,get_model_path("{}.pt".format(filename+'_beforeHardprune')))
    
    # hard prune
    hard_prune(admm, model, config_dict['sparsity_type'], option=None)
    if config_dict['sparsity_type']=='filter':
        test_filter_sparsity(model)
    else:
        test_irregular_sparsity(model)
    masks = get_model_mask(model=model)
    
    # masked retrain
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['retrain_bs'])
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader, \
                                         config_dict['retrain_opt'], config_dict['retrain_lr'], re_nepoch)
    for cepoch in range(0, re_nepoch+1):
        if cepoch > 0:
            start_time = time.time() 
            # you can also re-write hsic_train function
            if config_dict['retraining_type'] == 'hsictrain':
                #Train hsic model parameters with backprop + hsic with masks
                hsic_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict, masks=masks)
            elif config_dict['retraining_type'] == 'backprop':
                #Train hsic model parameters with backprop with masks
                standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict, masks=masks)
            else:
                raise ValueError("Unknown training type or not support [{}]".format(config_dict['retraining_type']))
            epoch_time.update(time.time()-start_time)
        
        epoch_log_dict, best = eval_and_save(config_dict, model, test_loader, epoch_log_dict, cepoch, nepoch, best, True)
               
    log_dict['epoch_log_dict'] = epoch_log_dict
    filename = "{}.npy".format(os.path.splitext(config_dict['model_file'])[0])
    save_logs(log_dict, get_log_filepath("{}".format(filename)))

    # Test pruning ratio
    test_irregular_sparsity(model)
    print('Overall training time is {:.2f}s.'.format(epoch_time.sum))
    
    return batch_log_list, epoch_log_dict