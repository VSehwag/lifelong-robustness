import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import warnings
warnings.simplefilter("ignore")
import foolbox

from attacks import *

def get_attack_vector(name, allparams):
    """
        Get Untargetd attack vectors
    """
    if name == "linf":
        attack_vector = lambda model, x, ytrue, ytarget: linf(model, x, ytrue, allparams)
    elif name == "l2":
        attack_vector = lambda model, x, ytrue, ytarget: l2(model, x, ytrue, allparams)
    elif name == "snow":
        params = getattr(allparams, "snow")
        attack = SnowAttack(nb_its=params.nb_its, eps_max=params.eps_max, step_size=params.step_size, 
                            resol=params.resol, rand_init=params.rand_init, scale_each=params.scale_each,
                            budget=params.budget)
        attack_vector = lambda model, x, ytrue, ytarget: (attack._forward(model, x, ytarget, avoid_target=True, scale_eps=False), ytrue)
    else:
        raise ValueError(f"{name} attack vector not supported")
    
    return attack_vector


def tr(model, x, y, allparams):
    params = getattr(allparams, "tr")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        attack = foolbox.attacks.SpatialAttack(params.max_translation, params.max_rotation, 
                                               params.num_translations, params.num_rotations, 
                                               params.grid_search, params.random_steps)
        _, xadv, _ = attack(foolbox.PyTorchModel(model, bounds=(params.clip_min, params.clip_max)), x, y)
    return xadv, y


def avg(model, x, y, allparams):
    xadv_linf, _ = linf(model, x, y, allparams)
    #xadv_l2, _ = l2(model, x, y, allparams)
    xadv_tr, _ = tr(model, x, y, allparams)
    return torch.cat([xadv_linf, xadv_tr]), torch.cat([y, y])
    

def max(model, x, y, allparams):
    xadv_linf, _ = linf(model, x, y, allparams)
    #xadv_l2, _ = l2(model, x, y, allparams)
    xadv_tr, _ = tr(model, x, y, allparams)
    
    conf_linf = F.softmax(model(xadv_linf), dim=-1).gather(dim=-1, index=y.view(-1, 1)).view(-1, 1, 1, 1)
    #conf_l2 = F.softmax(model(xadv_l2), dim=-1).gather(dim=-1, index=y.view(-1, 1)).view(-1, 1, 1, 1)
    conf_tr = F.softmax(model(xadv_tr), dim=-1).gather(dim=-1, index=y.view(-1, 1)).view(-1, 1, 1, 1)
    
    xadv = torch.where(conf_linf < conf_tr, xadv_linf, xadv_tr)
    
    return xadv, y
    