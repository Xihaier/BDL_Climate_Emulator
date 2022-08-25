'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import torch


def parameters_to_vector(parameters, grad=False, both=False):
    param_device = None
    if not both:
        vec = []
        if not grad:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.data.view(-1))
        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.data.view(-1))
        return torch.cat(vec)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)


def vector_to_parameters(vec, parameters, grad=True):
    param_device = None
    pointer = 0
    
    if grad:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(param.size())
            pointer += num_param
    else:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[pointer:pointer + num_param].view(param.size())
            pointer += num_param


def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, this is currently not supported.')
    return old_param_device


def log_sum_exp(input, dim=None, keepdim=False):
    if dim is None:
        input = input.view(-1)
        dim = 0
    max_val = input.max(dim=dim, keepdim=True)[0]
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output


