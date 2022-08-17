from .. import *
from .  import *

from ..math.admm import *
from ..model.lenet import *
from ..model.vgg import *
from ..model.resnet import *
from ..model.wideresnet import WideResNet28_4, WideResNet34_10


def activations_extraction(model, data_loader, out_dim=10, hid_idx=-1,):

    out_activation = np.zeros([len(data_loader)*data_loader.batch_size, out_dim])
    out_label = np.zeros([len(data_loader)*data_loader.batch_size,])
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(data_loader):
        
        if len(data)<data_loader.batch_size:
            break

        data = data.to(device)
        output, hiddens = model(data)
        
        begin = batch_idx*data_loader.batch_size
        end = (batch_idx+1)*data_loader.batch_size
        out_activation[begin:end] = hiddens[hid_idx].detach().cpu().numpy()
        out_label[begin:end] = target.detach().cpu().numpy()
        
    return {"activation":out_activation, "label":out_label}


def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):


    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)

    return hsic_hx_val, hsic_hy_val

def set_optimizer(config_dict, model, train_loader, opt, lr, epochs):
    """ bag of tricks set-ups"""
    config_dict['smooth'] = config_dict['smooth_eps'] > 0.0
    #config_dict['mixup'] = config_dict['alpha'] > 0.0

    optimizer_init_lr = config_dict['warmup_lr'] if config_dict['warmup'] else lr
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9,weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
    '''if config_dict['training_type'] == 'competitor3' or config_dict['training_type'] == 'competitor4' or config_dict['adv_train']:
        if config_dict['data_code'] == 'mnist':
            print('using sgd as optimizer')
            optimizer = optim.SGD(model.parameters(), lr=config_dict['learning_rate'], momentum=0.9)
        elif config_dict['data_code'] == 'cifar10':
            optimizer = optim.SGD(model.parameters(), lr=config_dict['learning_rate'], momentum=0.9, weight_decay=0.0035)
    '''   
    scheduler = None
    if config_dict['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=4e-08)
    else:
        """Set the learning rate of each parameter group to the initial lr decayed
                by gamma once the number of epoch reaches one of the milestones
        """
        gamma=0.1
        if config_dict['data_code'] == 'mnist':
            if epochs <= 50:
                epoch_milestones = [20, 40]
            else:
                epoch_milestones = [75, 90]
        elif config_dict['data_code'] == 'cifar10':
            if epochs > 150:
                epoch_milestones = [80, 150]
            else:
                epoch_milestones = [65, 90]
        elif config_dict['data_code'] == 'cifar100':
            # Adversarial Concurrent Training: Optimizing Robustness and Accuracy Trade-off of Deep Neural Networks
            if epochs > 150:
                epoch_milestones = [80, 150]
            else:
                epoch_milestones = [65, 90]
            
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=gamma)
        
    if config_dict['warmup']:
        scheduler = GradualWarmupScheduler(optimizer,multiplier=config_dict['learning_rate']/config_dict['warmup_lr'],
                                           total_iter=config_dict['warmup_epochs'] * len(train_loader),
                                           after_scheduler=scheduler)
        
    return optimizer, scheduler

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv)[0], y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)[0]
    logits_adv = model(x_adv)[0]

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss, logits

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    device = next(model.parameters()).device
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv)[0], dim=1),
                                       F.softmax(model(x_natural)[0], dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv)[0], dim=1),
                                           F.softmax(model(x_natural)[0], dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)[0]
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv)[0], dim=1),
                                                    F.softmax(model(x_natural)[0], dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, logits.detach(), model(x_adv)[1]

def model_distribution(config_dict):

    if config_dict['model'] == 'lenet3':
        model = LeNet3(**config_dict)
    elif config_dict['model'] == 'lenet4':
        model = LeNet4(**config_dict)
    elif config_dict['model'] == 'vgg16':
        model = VGG16(**config_dict)
    elif config_dict['model'] == 'resnet18':
        model = ResNet18(**config_dict)
    elif config_dict['model'] == 'resnet50':
        model = ResNet50(**config_dict)
    elif config_dict['model'] == 'wideresnet' or config_dict['model'] == 'wrn28-10':
        model = WideResNet28_10(**config_dict)
    elif config_dict['model'] == 'wrn28-4':
        model = WideResNet28_4(**config_dict)
    elif config_dict['model'] == 'wrn34-10':
        model = WideResNet34_10(**config_dict)
    else:
        raise ValueError("Unknown model name or not support [{}]".format(config_dict['model']))

    return model
