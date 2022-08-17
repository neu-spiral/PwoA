from .. import *
from .  import *
from .train_misc     import *
from ..math.admm import *
from copy import deepcopy
import torch.nn.functional as F

def hsic_prune(cepoch, model, data_loader, optimizer, scheduler, config_dict, ADMM=None, masks=None):

    prec1 = total_loss = hx_l = hy_l = -1

    batch_acc    = meter.AverageMeter()
    batch_loss   = meter.AverageMeter()
    batch_hsicloss   = meter.AverageMeter()
    batch_totalloss   = meter.AverageMeter()
    batch_hischx = meter.AverageMeter()
    batch_hischy = meter.AverageMeter()

    model = model.to(config_dict['device'])
    model.train()
    
    batch_size = config_dict['retrain_bs'] if masks is not None else config_dict['batch_size']
    n_data = batch_size * len(data_loader)

    ### initialize admm
    if ADMM is not None: 
        admm_initialization(config_dict, ADMM=ADMM, model=model)
    
    ### initialize teacher model
    if config_dict['distill']:
        pretrained = config_dict['pretrained']
        distillation_criterion = config_dict['distillation_criterion']
        batch_distillation_loss = meter.AverageMeter()


    if config_dict['adv_train']:
        if config_dict['attack_type'] == 'aa':
            attack = torchattacks.AutoAttack(model, norm='Linf', eps=config_dict['epsilon'], version='only2', 
                                             n_classes=config_dict['num_classes'], seed=None, verbose=False)
        elif config_dict['attack_type'] == 'pgd':
            attack = torchattacks.PGD(model,
                                      eps=config_dict['epsilon'],
                                      alpha=config_dict['pgd_alpha'],
                                      steps=config_dict['pgd_steps'],
                                      random_start=config_dict['random_start'],
                                      )
    start_time = time.time()   
    pbar = tqdm(enumerate(data_loader), total=n_data/batch_size, ncols=200)
    for batch_idx, (data, target) in pbar:
        
        # data augmentation for robustness attack
        if config_dict['aug_data']:
            attack = np.random.uniform(-config_dict['epsilon'], config_dict['epsilon'], size=data.shape)
            data += attack
            
        data   = data.float().to(config_dict['device'])
        target = target.to(config_dict['device'])
        hsic_loss, total_loss = 0, 0
        optimizer.zero_grad()
        flag_adv = False
        
        # several tricks
        if config_dict['mix_up']:
            data, target_a, target_b, lam = mixup_data(data, target, config_dict['alpha'])
        
        # adversarial training
        if config_dict['adv_train'] == 'adv':
            ##### Random select batch
            if np.random.random_sample() < config_dict['mix_ratio']:
                flag_adv = True
                # Trades generate adversarial examples inside
                if config_dict['base_loss'] == 'ce':
                    data = attack(data, target)
        # randomized smoothing
        elif config_dict['adv_train'] == 'smooth':
            data = data + torch.randn_like(data, device=config_dict['device']) * 0.25
                   
        # compute loss
        if config_dict['xentropy_weight'] > 0:
            if config_dict['base_loss'] == 'trades' and flag_adv:
                loss, output, hiddens = trades_loss(model, data, target, optimizer, beta=6.0)
            else:
                output, hiddens = model(data)
                criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config_dict['smooth_eps']).to(config_dict['device'])
                if config_dict['mix_up']:
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam, config_dict['smooth'])
                else:
                    loss = criterion(output, target, smooth=config_dict['smooth'])
            total_loss += (loss * config_dict['xentropy_weight'])
        else:
            output, hiddens = model(data)
            
        # compute hsic
        if masks is not None:
            lx, ly = config_dict['retrain_lx'], config_dict['retrain_ly']
        else:
            lx, ly = config_dict['lambda_x'], config_dict['lambda_y']
        
        if lx > 0 and ly > 0:
            h_target = target.view(-1,1)
            h_target = misc.to_categorical(h_target, num_classes=config_dict['num_classes']).float()
            h_data = data.view(-1, np.prod(data.size()[1:]))

            # Different here, we jointly optimize all HSICs together
            # We can have different weights for each layers

            # new variable
            hx_l_list = []
            hy_l_list = []

            for i in range(len(hiddens)):
                if len(hiddens[i].size()) > 2:
                    hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))
                hx_l, hy_l = hsic_objective(
                        hiddens[i],
                        h_target=h_target.float(),
                        h_data=h_data,
                        sigma=config_dict['sigma'],
                        k_type_y=config_dict['k_type_y']
                )
                hx_l_list.append(hx_l)
                hy_l_list.append(hy_l)

                temp_hsic = lx * hx_l - ly * hy_l
                hsic_loss += temp_hsic.to(config_dict['device'])

            total_loss += hsic_loss
                         
        ### add admm      
        if ADMM is not None:
            z_u_update(config_dict, ADMM, model, cepoch, batch_idx)  # update Z and U variables
            prev_loss, admm_loss, total_loss = append_admm_loss(ADMM, model, total_loss)  # append admm losses
        
        ### self-distillation
        if config_dict['distill']:
            output_pre, hiddens_pre = pretrained(data)
            if config_dict['distill_loss'] == 'kl':
                if config_dict['model'] == 'resnet18' or 'wideresnet':
                    dis_loss = config_dict['distill_temp'] * config_dict['distill_temp'] * distillation_criterion(F.log_softmax(output / config_dict['distill_temp'], dim=1), F.log_softmax(output_pre / config_dict['distill_temp'], dim=1))
                elif config_dict['model'] == 'lenet3':
                    dis_loss = config_dict['distill_temp'] * config_dict['distill_temp'] * distillation_criterion(output / config_dict['distill_temp'], output_pre / config_dict['distill_temp'])
                else:
                    raise NotImplementedError()
            elif config_dict['distill_loss'] == 'mseml':
                dis_loss = 0
                for q in range(len(hidden_pruned)):
                    dis_loss += distillation_criterion(hiddens[q], hiddens_pre[q])
            else:
                dis_loss = distillation_criterion(output, output_pre)
            alpha = config_dict['distill_alpha']
            total_loss += alpha * dis_loss
        
        total_loss.backward() # Back Propagation
        
        ### for masked training
        if masks is not None:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and W.grad is not None:
                        W.grad *= masks[name]
                        
        optimizer.step() # Gradient Descent
        # adjust learning rate
        if ADMM is not None:
            admm_adjust_learning_rate(optimizer, cepoch, config_dict)
        else:
            scheduler.step()
            
        prec1, prec5 = misc.get_accuracy(output, target, topk=(1, 5)) 

        batch_acc.update(prec1)
        batch_totalloss.update(float(total_loss.detach().cpu().numpy())) # admm loss
        if lx > 0 and ly > 0:
            batch_hsicloss.update(float(hsic_loss.detach().cpu().numpy())) # hsic loss
            batch_hischx.update(sum(hx_l_list).cpu().detach().numpy())
            batch_hischy.update(sum(hy_l_list).cpu().detach().numpy())
        else:
            batch_hsicloss.update(0) # hsic loss
            batch_hischx.update(0)
            batch_hischy.update(0)
            
        if config_dict['distill']:
            batch_distillation_loss.update(dis_loss.detach().cpu().numpy())

        # # # preparation log information and print progress # # #
        if config_dict['distill']:
            msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] hsicLoss: {hsic_loss:.4f} distillLoss: {distill_loss:.8f} totalLoss: {total_loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.2f} hsic_yz:{hsic_zy:.2f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        hsic_loss = batch_hsicloss.avg,
                        distill_loss = batch_distillation_loss.avg,
                        total_loss = batch_totalloss.avg,
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )
        else:
            msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] hsicLoss: {hsic_loss:.4f} totalLoss: {total_loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.4f} hsic_yz:{hsic_zy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        total_loss = batch_totalloss.avg,
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )

        pbar.set_description(msg)
    print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))