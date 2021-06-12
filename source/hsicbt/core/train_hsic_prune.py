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

    ### Tong: initialize admm
    if ADMM is not None: 
        admm_initialization(config_dict, ADMM=ADMM, model=model)
    
    ### Zifeng: initialize teacher model
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
    pbar = tqdm(enumerate(data_loader), total=n_data/batch_size, ncols=120)
    for batch_idx, (data, target) in pbar:
        
        # data augmentation for robustness attack
        if config_dict['aug_data']:
            attack = np.random.uniform(-config_dict['epsilon'], config_dict['epsilon'], size=data.shape)
            data += attack
            
        data   = data.float().to(config_dict['device'])
        target = target.to(config_dict['device'])
        total_loss = 0
        hsic_loss = 0
        optimizer.zero_grad()
        
        # several tricks
        if config_dict['mixup']:
            data, target_a, target_b, lam = mixup_data(data, target, config_dict['alpha'])
        
        # adversarial training
        if config_dict['adv_train'] == 'adv':
            ##### Random select batch
            if np.random.random_sample() < config_dict['mix_ratio']:
                data = attack(data, target)
            output, hiddens = model(data)
            
            '''
            ##### Random select samples in batch
            attacked_data = attack(data, target)
            if config_dict['mix_ratio'] > 0:
                bs = attacked_data.shape[0]
                indices = np.random.choice(np.arange(bs), size=int(np.ceil(bs*config_dict['mix_ratio'])), replace=False)
                data[indices] = attacked_data[indices]
                attacked_data = data
            output, hiddens = model(attacked_data)
            data = attacked_data
            '''
        elif config_dict['adv_train'] == 'nat':
            attacked_data = attack(data, target)
            output, _ = model(attacked_data)
            _, hiddens = model(data)
        else:
            output, hiddens = model(data)
        
        # compute loss
        criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config_dict['smooth_eps']).to(config_dict['device'])
        if config_dict['mixup']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam, config_dict['smooth'])
        else:
            loss = criterion(output, target, smooth=config_dict['smooth'])
        total_loss += (loss * config_dict['xentropy_weight'])
        
        # compute hsic
        h_target = target.view(-1,1)
        h_target = misc.to_categorical(h_target, num_classes=config_dict['num_classes']).float()
        h_data = data.view(-1, np.prod(data.size()[1:]))


        # Zifeng: Different here, we jointly optimize all HSICs together
        # We can have different weights for each layers
        
        # new variable
        hx_l_list = []
        hy_l_list = []
        
        if masks is not None:
            lx, ly, ld = config_dict['retrain_lx'], config_dict['retrain_ly'], config_dict['hsic_layer_decay']
        else:
            lx, ly, ld = config_dict['lambda_x'], config_dict['lambda_y'], config_dict['hsic_layer_decay']
            
        if ld > 0:
            lx, ly = lx * (ld ** len(hiddens)), ly * (ld ** len(hiddens))
        
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
            
            if ld > 0:
                lx, ly = lx/ld, ly/ld
                #print(i, lx, ly)
            temp_hsic = lx * hx_l - ly * hy_l
            hsic_loss += temp_hsic.to(config_dict['device'])
        total_loss += hsic_loss
                         
        ### Tong: add admm                  
        if ADMM is not None:
            z_u_update(config_dict, ADMM, model, cepoch, batch_idx)  # update Z and U variables
            prev_loss, admm_loss, total_loss = append_admm_loss(ADMM, model, total_loss)  # append admm losses
        
        ### Zifeng
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
        
        
        ### Tong: for masked training
        if masks is not None:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
                        
        optimizer.step() # Gradient Descent
        # adjust learning rate
        if ADMM is not None:
            admm_adjust_learning_rate(optimizer, cepoch, config_dict)
        else:
            scheduler.step()
            
        prec1, prec5 = misc.get_accuracy(output, target, topk=(1, 5)) 

        batch_acc.update(prec1)
        batch_loss.update(float(loss.detach().cpu().numpy())) # this is just for xentropy loss! total loss is for xentropy + hsic
        batch_hsicloss.update(float(hsic_loss.detach().cpu().numpy())) # hsic loss
        batch_totalloss.update(float(total_loss.detach().cpu().numpy())) # admm loss
        batch_hischx.update(sum(hx_l_list).cpu().detach().numpy())
        batch_hischy.update(sum(hy_l_list).cpu().detach().numpy())

        if config_dict['distill']:
            batch_distillation_loss.update(dis_loss.detach().cpu().numpy())
        #batch_hischx.update(hx_l_list[-1].cpu().detach().numpy())
        #batch_hischy.update(hy_l_list[-1].cpu().detach().numpy())

        # # # preparation log information and print progress # # #
        if config_dict['distill']:
            msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} hsicLoss: {hsic_loss:.4f} distillLoss: {distill_loss:.4f} totalLoss: {total_loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.4f} hsic_yz:{hsic_zy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        loss = batch_loss.avg, 
                        hsic_loss = batch_hsicloss.avg,
                        distill_loss = batch_distillation_loss.avg,
                        total_loss = batch_totalloss.avg,
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )
        else:
            msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} hsicLoss: {hsic_loss:.4f} totalLoss: {total_loss:.4f} Acc:{acc:.4f} hsic_xz:{hsic_zx:.4f} hsic_yz:{hsic_zy:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*config_dict['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*config_dict['batch_size']/n_data), 
                        loss = batch_loss.avg, 
                        hsic_loss = batch_hsicloss.avg,
                        total_loss = batch_totalloss.avg,
                        acc  = batch_acc.avg,
                        hsic_zx = batch_hischx.avg,
                        hsic_zy = batch_hischy.avg,
                    )

        pbar.set_description(msg)
    print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))