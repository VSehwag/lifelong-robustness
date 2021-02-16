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
        Get Untargetd attack vectors. 
        Some attack expect model to work on [0, 255] pixel range, where existing input and model have [0, 1] pixel range. Using a wrapper around current models to handle this. Note that the wrapper model no longer an instance of nn.Module class.
    """
    if name == "none":
        attack_vector = lambda model, x, ytrue, ytarget: (x, ytrue)
    elif name == "linf":
        attack_vector = lambda model, x, ytrue, ytarget: linf(model, x, ytrue, allparams)
    elif name == "l2":
        attack_vector = lambda model, x, ytrue, ytarget: l2(model, x, ytrue, allparams)
    elif name == "snow":
        params = getattr(allparams, "snow")
        attack = SnowAttack(nb_its=params.nb_its, eps_max=params.eps_max, step_size=params.step_size, 
                            resol=params.resol, rand_init=params.rand_init, scale_each=params.scale_each,
                            budget=params.budget)
        attack_vector = lambda model, x, ytrue, ytarget: (attack._forward(lambda x: model(x/255.), x, ytarget, avoid_target=True, scale_eps=False), ytrue)
    elif name == "gabor":
        params = getattr(allparams, "gabor")
        attack = GaborAttack(nb_its=params.nb_its, eps_max=params.eps_max, step_size=params.step_size, 
                            resol=params.resol, rand_init=params.rand_init, scale_each=params.scale_each)
        attack_vector = lambda model, x, ytrue, ytarget: (attack._forward(lambda x: model(x/255.), x, ytarget, avoid_target=True, scale_eps=False), ytrue)
    elif name == "jpeg":
        params = getattr(allparams, "jpeg")
        attack = JPEGAttack(nb_its=params.nb_its, eps_max=params.eps_max, step_size=params.step_size, 
                            resol=params.resol, rand_init=params.rand_init, scale_each=params.scale_each, opt=params.opt)
        attack_vector = lambda model, x, ytrue, ytarget: (attack._forward(lambda x: model(x/255.), x, ytarget, avoid_target=True, scale_eps=False), ytrue)
    else:
        raise ValueError(f"{name} attack vector not supported")
    
    return attack_vector


def combine_attack_vectors(names, allparams, mode="max"):
    assert mode in ["max", "avg"]
    attack_vectors = [get_attack_vector(name, allparams) for name in names]
    print(f"Combining {names} attacks with {mode} mode")
    if mode == "avg":
        def attack_vector(model, x, ytrue, ytarget):
            xadv = torch.cat([attack(model, x, ytrue, ytarget)[0] for attack in attack_vectors]) # all returns the original label itself
            y = torch.cat([ytrue, ytrue])
            return xadv, y
    if mode =="max":
        assert len(names) == 2, "currently supporting averaging of two attacks only, need a more generic comparison fxn to support more attacks"
        def attack_vector(model, x, ytrue, ytarget):
            xadv = [attack(model, x, ytrue, ytarget)[0] for attack in attack_vectors] # all returns the original label itself
            pred = [F.softmax(model(v), dim=-1).gather(dim=-1, index=ytrue.view(-1, 1)).view(-1, 1, 1, 1) for v in xadv]
            xadv = torch.where(pred[0] < pred[1], xadv[0], xadv[1])
            return xadv, ytrue
    return attack_vector