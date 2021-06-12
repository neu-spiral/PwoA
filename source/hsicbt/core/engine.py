import collections

from .. import *
from .  import *

from .train_misc     import *
from .train_hsic     import *
from .train_standard import *
from ..utils.path    import *

from ..math.admm import *
from ..utils.masks import *
from .train_competitor1 import *
from .train_competitor2 import *
from .train_competitor3 import *
from .train_competitor4 import *
from .train_hsic_prune     import *
from .train_hsic_prune_proj     import *

import torch

def training_standard(config_dict):
    """
    Train model with HBaR or CE only
    """
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['batch_size'])

    torch.manual_seed(config_dict['seed'])

    model = model_distribution(config_dict)
    #config_dict['robustness'] = True
    model_single_output = model_distribution(config_dict)
    for name, weight in model.named_parameters():
        print(name)

    # load pretrained model
    if 'load_model' in config_dict:
        model.load_state_dict(load_model(get_model_path("{}".format(config_dict['load_model']))))
        
    nepoch = config_dict['epochs']
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader, \
                                         config_dict['optimizer'], config_dict['learning_rate'], nepoch)

    best = 0
    log_dict = {}
    batch_log_list = []
    epoch_log_dict = collections.defaultdict(list)

    
    filename = os.path.splitext(config_dict['model_file'])[0]
    #train_loss, train_acc, train_hx, train_hy = misc.get_hsic_epoch(config_dict, model, train_loader)
    #test_loss, test_acc, test_hx, test_hy = misc.get_hsic_epoch(config_dict, model, test_loader)
    #epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
    #print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}.".format(0, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy))
    
    for cepoch in range(0, nepoch+1):
        if cepoch > 0:
            print(get_lr(optimizer))
            if config_dict['training_type'] == 'hsictrain':
                hsic_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            elif config_dict['training_type'] == 'backprop':
                standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            elif config_dict['training_type'] == 'competitor1': ### competing method: HSIC BottleNeck
                competitor1(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            elif config_dict['training_type'] == 'competitor2': ### competing method: HSIC(X, Y-h(x)) + CE
                competitor2(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            elif config_dict['training_type'] == 'competitor3': ### competing method: adversarial training via mart
                competitor3(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            elif config_dict['training_type'] == 'competitor4': ### competing method: adversarial training via trades
                competitor4(cepoch, model, train_loader, optimizer, scheduler, config_dict)
            else:
                raise ValueError("Unknown training type or not support [{}]".format(config_dict['training_type']))
            
        
        train_loss, train_acc, train_hx, train_hy, train_dl = misc.get_hsic_epoch(config_dict, model, train_loader)
        test_loss, test_acc, test_hx, test_hy, test_dl = misc.get_hsic_epoch(config_dict, model, test_loader)
        epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
        print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}.".format(cepoch, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy))
          
        # Robustness Analysis
        model_single_output.load_state_dict(model.state_dict())
        model_single_output.eval()
        rob_acc, rob_acc5, rob_hx, rob_hy = misc.eval_robust_epoch(model_single_output, test_loader, config_dict)
        epoch_log_dict['rob'].append(rob_acc)
        epoch_log_dict['rob_hx'].append(rob_hx)
        epoch_log_dict['rob_hy'].append(rob_hy)
        if config_dict['save_last_model_only'] and cepoch == nepoch:
            save_model(model,get_model_path("{}.pt".format(filename)))
        else:
            temp = test_acc + rob_acc
            if temp > best:
                save_model(model,get_model_path("{}.pt".format(filename)))
                best = temp
                
            
    log_dict['epoch_log_dict'] = epoch_log_dict
    log_dict['config_dict'] = config_dict
    filename = "{}.npy".format(os.path.splitext(config_dict['model_file'])[0])
    save_logs(log_dict, get_log_filepath("{}".format(filename)))

    return batch_log_list, epoch_log_dict

def training_hsic_prune(config_dict):
    """
    Train hsic model parameters with hsic + hard pruning + masked retrain
    """
    
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['batch_size'])
    torch.manual_seed(config_dict['seed'])
    
    # load pre-trained model
    model = model_distribution(config_dict)
    model.load_state_dict(load_model(get_model_path("{}".format(config_dict['load_model']))))

    # distillation teacher model
    if config_dict['distill']:
        pretrained = deepcopy(model)
        pretrained.load_state_dict(torch.load(config_dict['distill_model_path']))
        pretrained = pretrained.to(config_dict['device'])
        pretrained.eval()
        config_dict['pretrained'] = pretrained
        if config_dict['distill_loss'] == 'kl':
            distillation_criterion = torch.nn.KLDivLoss(log_target=True).to(config_dict['device'])
        elif config_dict['distill_loss'] == 'mse' or 'mseml':
            distillation_criterion = torch.nn.MSELoss().to(config_dict['device'])
        config_dict['distillation_criterion'] = distillation_criterion

    # construct single output model for test
    # config_dict['robustness'] = True
    model_single_output = model_distribution(config_dict)
    
    # optimizer
    nepoch = config_dict['epochs']
    re_nepoch = config_dict['retrain_ep']
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader, \
                                         config_dict['optimizer'], config_dict['learning_rate'], nepoch)
    
    log_dict = {}
    batch_log_list = []
    epoch_log_dict = collections.defaultdict(list)
    
    
    # pre-test
    train_loss, train_acc, train_hx, train_hy, train_dl = misc.get_hsic_epoch(config_dict, model, train_loader)
    test_loss, test_acc, test_hx, test_hy, test_dl = misc.get_hsic_epoch(config_dict, model, test_loader)
    epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
    print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, train_dl {:.4f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, test_dl {:.4f}.".format(0, train_acc, train_hx, train_hy, train_dl, test_loss, test_acc, test_hx, test_hy, test_dl))
    
    # Robustness Analysis
    model_single_output.load_state_dict(model.state_dict())
    model_single_output.eval()
    rob_acc, rob_acc5, rob_hx, rob_hy = misc.eval_robust_epoch(model_single_output, test_loader, config_dict)
    epoch_log_dict['rob'].append(rob_acc)
    epoch_log_dict['rob_hx'].append(rob_hx)
    epoch_log_dict['rob_hy'].append(rob_hy)
    
    # Tong： Initializing ADMM; if not admm, do hard pruning only
    admm = ADMM(config_dict, model, config=get_pr_path(config_dict), rho=config_dict['rho']) if config_dict['admm'] else None
    
    
    for cepoch in range(1, nepoch+1):
        if config_dict['training_type'] == 'hsicproj':
            #Train hsic model parameters with backprop + pruning, in the end of each epoch, do ADMM + hsic projection
            hsic_proj_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict, ADMM=admm)
        elif config_dict['training_type'] == 'hsictrain':
            #Train hsic model parameters with (backprop + hisc) + pruning, in the end of each epoch, do ADMM
            hsic_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict, ADMM=admm)
        elif config_dict['training_type'] == 'backprop':
            #Train hsic model parameters with backprop + pruning, in the end of each epoch, do ADMM
            standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict, ADMM=admm)
        else:
            raise ValueError("Unknown training type or not support [{}]".format(config_dict['training_type']))

        train_loss, train_acc, train_hx, train_hy, train_dl = misc.get_hsic_epoch(config_dict, model, train_loader)
        test_loss, test_acc, test_hx, test_hy, test_dl = misc.get_hsic_epoch(config_dict, model, test_loader)
        epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
        print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, train_dl {:.4f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, test_dl {:.4f}.".format(cepoch, train_acc, train_hx, train_hy, train_dl, test_loss, test_acc, test_hx, test_hy, test_dl))

        # Robustness Analysis
        model_single_output.load_state_dict(model.state_dict())
        model_single_output.eval()
        rob_acc, rob_acc5, rob_hx, rob_hy = misc.eval_robust_epoch(model_single_output, test_loader, config_dict)
        epoch_log_dict['rob'].append(rob_acc)
        epoch_log_dict['rob_hx'].append(rob_hx)
        epoch_log_dict['rob_hy'].append(rob_hy)
        print(get_lr(optimizer))
    
    # Tong： if not admm, do hard pruning only
    
    #filename = os.path.splitext(config_dict['model_file'])[0]
    #save_model(model,get_model_path("{}.pt".format(filename+'_beforeHardprune')))
       
    # hard prune
    hard_prune(admm, model, config_dict['sparsity_type'], option=None)
    if config_dict['sparsity_type']=='filter':
        test_filter_sparsity(model)
    else:
        test_irregular_sparsity(model)
    masks = get_model_mask(model=model)
    train_loss, train_acc, train_hx, train_hy, train_dl = misc.get_hsic_epoch(config_dict, model, train_loader)
    test_loss, test_acc, test_hx, test_hy, test_dl = misc.get_hsic_epoch(config_dict, model, test_loader)
    epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
    # print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, train_dl {:.4f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, test_dl {:.4f}.".format(0, train_acc, train_hx, train_hy, train_dl, test_loss, test_acc, test_hx, test_hy, test_dl))
    print("After hard prune: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, train_dl {:.4f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, test_dl {:.4f}.".format(train_acc, train_hx, train_hy, train_dl, test_loss, test_acc, test_hx, test_hy, test_dl))
    # Robustness Analysis
    model_single_output.load_state_dict(model.state_dict())
    model_single_output.eval()
    rob_acc, rob_acc5, rob_hx, rob_hy = misc.eval_robust_epoch(model_single_output, test_loader, config_dict)
    epoch_log_dict['rob'].append(rob_acc)
    epoch_log_dict['rob_hx'].append(rob_hx)
    epoch_log_dict['rob_hy'].append(rob_hy)

    # masked retrain
    best = 0
    train_loader, test_loader = get_dataset_from_code(config_dict['data_code'], config_dict['retrain_bs'])
    optimizer, scheduler = set_optimizer(config_dict, model, train_loader, \
                                         config_dict['retrain_opt'], config_dict['retrain_lr'], re_nepoch)
    for cepoch in range(1, re_nepoch+1):
        # you can also re-write hsic_train function
        if config_dict['retraining_type'] == 'hsicproj' or config_dict['retraining_type'] == 'hsictrain':
            #Train hsic model parameters with backprop + hsic with masks
            hsic_prune(cepoch, model, train_loader, optimizer, scheduler, config_dict, masks=masks)
        elif config_dict['retraining_type'] == 'backprop':
            #Train hsic model parameters with backprop with masks
            standard_train(cepoch, model, train_loader, optimizer, scheduler, config_dict, masks=masks)
        else:
            raise ValueError("Unknown training type or not support [{}]".format(config_dict['retraining_type']))
            
        train_loss, train_acc, train_hx, train_hy, train_dl = misc.get_hsic_epoch(config_dict, model, train_loader)
        test_loss, test_acc, test_hx, test_hy, test_dl = misc.get_hsic_epoch(config_dict, model, test_loader)
        epoch_log_dict = append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy)
        print("Epoch-[{:03d}]: Train acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, train_dl {:.4f}; Test loss: {:.2f}, acc: {:.2f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}, test_dl {:.4f}.".format(cepoch, train_acc, train_hx, train_hy, train_dl, test_loss, test_acc, test_hx, test_hy, test_dl))
        
        
        
        # Robustness Analysis
        model_single_output.load_state_dict(model.state_dict())
        model_single_output.eval()
        rob_acc, rob_acc5, rob_hx, rob_hy = misc.eval_robust_epoch(model_single_output, test_loader, config_dict)
        epoch_log_dict['rob'].append(rob_acc)
        epoch_log_dict['rob_hx'].append(rob_hx)
        epoch_log_dict['rob_hy'].append(rob_hy)
        print(get_lr(optimizer))
        
        # Save Model
        #temp = test_acc + rob_acc
        temp = rob_acc
        if temp > best:
            filename = os.path.splitext(config_dict['model_file'])[0]
            print("!!!!!Saving the model at retrain epoch {}".format(cepoch))
            save_model(model,get_model_path("{}.pt".format(filename+'_best')))
            best = temp
        #if config_dict['save_last_model_only'] and cepoch == re_nepoch:
        #    filename = os.path.splitext(config_dict['model_file'])[0]
        #    save_model(model,get_model_path("{}.pt".format(filename)))
            
        
                
    log_dict['epoch_log_dict'] = epoch_log_dict
    #log_dict['config_dict'] = config_dict
    filename = "{}.npy".format(os.path.splitext(config_dict['model_file'])[0])
    save_logs(log_dict, get_log_filepath("{}".format(filename)))

    # Test pruning ratio
    test_irregular_sparsity(model)
    
    return batch_log_list, epoch_log_dict