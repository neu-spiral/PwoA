from .. import *
from ..math.hsic import *
import torchattacks

def get_current_timestamp():
    return strftime("%y%m%d_%H%M%S")

def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'cifar10':
        in_ch = 3
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

def get_in_dimensions(data_code):
    in_dim = -1    
    if data_code == 'mnist':
        in_dim = 784
    elif data_code == 'cifar10':
        in_dim = 1024
    elif data_code == 'fmnist':
        in_dim = 784
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_dim

def get_accuracy_epoch(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    acc = []
    loss = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = model.to('cuda')
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        output, hiddens = model(data)
        loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
        acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
    return np.mean(acc), np.mean(loss)

def get_hsic_epoch(config_dict, model, dataloader):
    """ Computes the hsic
    """
    acc = []
    loss = []
    hx_l_list = []
    hy_l_list = []
    model = model.to('cuda')
    device = next(model.parameters()).device
    
    model.eval()

    if config_dict['distill']:
        pretrained = config_dict['pretrained']
        distillation_criterion = config_dict['distillation_criterion']
        distill_l_list = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            output, hiddens = model(data)

            # compute acc
            acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())

            # compute hsic
            h_target = target.view(-1,1)
            h_target = to_categorical(h_target, num_classes=config_dict['num_classes']).float()
            h_data = data.view(-1, np.prod(data.size()[1:]))

            hsic_hx = hsic_normalized_cca( hiddens[-1], h_data,   sigma=config_dict['sigma'])
            hsic_hy = hsic_normalized_cca( hiddens[-1], h_target, sigma=config_dict['sigma'], k_type_y=config_dict['k_type_y'])
            hsic_hx = hsic_hx.cpu().detach().numpy()
            hsic_hy = hsic_hy.cpu().detach().numpy()

            hx_l_list.append(hsic_hx)
            hy_l_list.append(hsic_hy)

            # compute loss
            loss.append(torch.nn.CrossEntropyLoss()(output, target).cpu().detach().numpy())

            if config_dict['distill']:
                output_pre, hiddens_pre = pretrained(data)
                if config_dict['distill_loss'] == 'kl':
                    if config_dict['model'] == 'resnet18' or 'wideresnet':
                        dis_loss = config_dict['distill_temp'] * config_dict['distill_temp'] * distillation_criterion(F.log_softmax(output / config_dict['distill_temp'], dim=1), F.log_softmax(output_pre / config_dict['distill_temp'], dim=1))
                    else:
                        raise NotImplementedError()
                elif config_dict['distill_loss'] == 'mseml':
                    dis_loss = 0
                    for q in range(len(hidden_pruned)):
                        dis_loss += distillation_criterion(hiddens[q], hiddens_pre[q])
                else:
                    dis_loss = distillation_criterion(output, output_pre)
                distill_l_list.append(dis_loss.detach().cpu().numpy())
        if config_dict['distill']:
            distill_loss = np.mean(distill_l_list)
        else:
            distill_loss = -1          

    return np.mean(loss), np.mean(acc), np.mean(hx_l_list), np.mean(hy_l_list), distill_loss


def eval_robust_epoch(model, dataloader, config_dict):
    acc = []
    acc5 = []
    hx_l_list = []
    hy_l_list = []
    model = model.to('cuda')
    device = next(model.parameters()).device
    model.eval()
    
    eps = config_dict['epsilon']
    alpha = config_dict['pgd_alpha']
    if config_dict['attack_type'] == 'aa':
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=config_dict['epsilon'], version='standard', 
                                             n_classes=config_dict['num_classes'], seed=None, verbose=False)
    else:
        attack = torchattacks.PGD(model, 
                                  eps=eps, 
                                  alpha=alpha, 
                                  steps=config_dict['pgd_steps'],
                                  random_start=config_dict['random_start'])
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)

        attacked_data = attack(data, target)
        output, hiddens = model(attacked_data)

        prec1, prec5 = get_accuracy(output, target, topk=(1, 5)) 
        acc.append(prec1.cpu().detach().numpy())
        acc5.append(prec5.cpu().detach().numpy())
        
        # compute hsic
        h_target = target.view(-1,1)
        h_target = to_categorical(h_target, num_classes=config_dict['num_classes']).float()
        h_data = data.view(-1, np.prod(data.size()[1:]))

        hsic_hx = hsic_normalized_cca( hiddens[-1], h_data,   sigma=config_dict['sigma'])
        hsic_hy = hsic_normalized_cca( hiddens[-1], h_target, sigma=config_dict['sigma'], k_type_y=config_dict['k_type_y'])
        hsic_hx = hsic_hx.cpu().detach().numpy()
        hsic_hy = hsic_hy.cpu().detach().numpy()

        hx_l_list.append(hsic_hx)
        hy_l_list.append(hsic_hy)
        
        if config_dict['attack_type'] == 'aa':
            print('For AutoAttack, only evaluate one batch of the data')
            # print(data[0])
            # print(target[0])
            # print(attacked_data[0])
            # print(output[0])
            # print(torch.max(torch.abs(attacked_data[0]-data[0])))
            # # print()
            # exit(0)
            break
    print("Average robust accuracy is top1: {:.4f}, top5: {:.4f}, hsic_xz: {:.2f}, hsic_yz: {:.2f}.".format(np.mean(acc), np.mean(acc5), np.mean(hx_l_list), np.mean(hy_l_list)))
    return np.mean(acc), np.mean(acc5), np.mean(hx_l_list), np.mean(hy_l_list)

def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_accuracy_hsic(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    for batch_idx, (data, target) in enumerate(dataloader):
        output, hiddens = model(data.to(next(model.parameters()).device))
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy().reshape(-1,1)
        output_list.append(output)
        target_list.append(target)
    output_arr = np.vstack(output_list)
    target_arr = np.vstack(target_list)
    avg_acc = 0
    reorder_list = []
    for i in range(10):
        indices = np.where(target_arr==i)[0]
        select_item = output_arr[indices]
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        while np.argmax(y) in reorder_list:
            y[np.argmax(y)] = 0
        reorder_list.append(np.argmax(y))
        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0])/float(out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.

    return avg_acc*100., reorder_list

def append_epoch_log_dict(epoch_log_dict, train_loss, train_acc, train_hx, train_hy, test_loss, test_acc, test_hx, test_hy):
    epoch_log_dict['train_loss'].append(train_loss)
    epoch_log_dict['train_acc'].append(train_acc)
    epoch_log_dict['train_hx'].append(train_hx)
    epoch_log_dict['train_hy'].append(train_hy)

    epoch_log_dict['test_loss'].append(test_loss)
    epoch_log_dict['test_acc'].append(test_acc)
    epoch_log_dict['test_hx'].append(test_hx)
    epoch_log_dict['test_hy'].append(test_hy)
    return epoch_log_dict

def get_layer_parameters(model, idx_range):

    param_out = []
    param_out_name = []
    for it, (name, param) in enumerate(model.named_parameters()):
        if it in idx_range:
            param_out.append(param)
            param_out_name.append(name)

    return param_out, param_out_name


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y])
