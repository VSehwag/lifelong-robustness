import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
#import foolbox

## adapted from https://github.com/yaodongyu/TRADES/blob/master/trades.py
def linf(model, x, y, allparams):
    params = getattr(allparams, "linf")
    device = x.device
    
    random_noise = torch.FloatTensor(x.shape).uniform_(-params.epsilon, params.epsilon).to(device).detach()
    xadv = Variable(x.detach().data + random_noise, requires_grad=True)
    
    for _ in range(params.steps):
        #print(torch.min(xadv).item(), torch.max(xadv).item())
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(xadv), y)
        loss.backward()
        xadv.data = xadv.data + params.step_size * xadv.grad.data.sign()
        eta = torch.clamp(xadv.data - x.data, -params.epsilon, params.epsilon)
        xadv.data = torch.clamp(x.data + eta, params.clip_min, params.clip_max)
        xadv.grad.data = torch.zeros_like(xadv.grad.data)  # zero out accumulated gradients
        
#     print("Linf = ", torch.max(xadv - x).abs().item(), 
#           "L-2 = ", torch.mean(torch.norm((xadv - x).view(len(x), -1), dim=-1)).item(), 
#           torch.min(xadv).item(), torch.max(xadv).item(), 
#           torch.min(x).item(), torch.max(x).item()
#          )
    return xadv, y


def l2(model, x, y, allparams):
    params = getattr(allparams, "l2")
    device = x.device
    
    random_noise = torch.FloatTensor(x.shape).uniform_(-1, 1).to(device).detach()
    random_noise.renorm_(p=2, dim=0, maxnorm=params.epsilon)
    
    xadv = Variable(x.detach().data + random_noise, requires_grad=True)
    
    for _ in range(params.steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(xadv), y)
        loss.backward()
        grad_norms = xadv.grad.view(len(x), -1).norm(p=2, dim=1)
        xadv.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            xadv.grad[grad_norms == 0] = torch.randn_like(xadv.grad[grad_norms == 0])
        
        xadv.data += params.step_size * xadv.grad.data
        eta = xadv.data - x.data
        eta.renorm_(p=2, dim=0, maxnorm=params.epsilon)
        xadv.data = torch.clamp(x.data + eta, params.clip_min, params.clip_max)
        xadv.grad.data = torch.zeros_like(xadv.grad.data)  # zero out accumulated gradients

#     print("Linf = ", torch.max(xadv - x).abs().item(), 
#           "L-2 = ", torch.mean(torch.norm((xadv - x).view(len(x), -1), dim=-1)).item(), 
#           torch.min(xadv).item(), torch.max(xadv).item(), 
#           torch.min(x).item(), torch.max(x).item()
#          )
    return xadv, y


def tr(model, x, y, allparams):
    # Using default +-3 pixel translation and +-30 degree rotation and assuming pixels in [0, 1] range.
    print("fds")
    attack = foolbox.attacks.SpatialAttack()
    print(attack)
    _, xadv, _ = attack(foolbox.PyTorchModel(model, bounds=(0, 1)), x, y)
    print(xadv.shape)
    return xadv, y


def avg(model, x, y, allparams):
    xadv_linf, _ = linf(model, x, y, allparams)
    xadv_l2, _ = l2(model, x, y, allparams)
    return torch.cat([xadv_linf, xadv_l2]), torch.cat([y, y])
    

def max(model, x, y, allparams):
    xadv_linf, _ = linf(model, x, y, allparams)
    xadv_l2, _ = l2(model, x, y, allparams)
    
    conf_linf = F.softmax(model(xadv_linf), dim=-1).gather(dim=-1, index=y.view(-1, 1)).view(-1, 1, 1, 1)
    conf_l2 = F.softmax(model(xadv_l2), dim=-1).gather(dim=-1, index=y.view(-1, 1)).view(-1, 1, 1, 1)

    xadv = torch.where(conf_linf < conf_l2, xadv_linf, xadv_l2)
    
    return xadv, y
    