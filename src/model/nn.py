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
import numpy as np

from model.BayesNN import BayesNet
from model.baselines import DenseNet, DenseNet11Conv, DenseNetFNN, ConvCrop, FNNCrop, ConvFNNCrop
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau


def getModel(args):
    if args.model == 'DenseNet':
        model = DenseNet(args.drop_out)
        model = model.to(args.device)
    elif args.model == 'DenseNet11':
        model = DenseNet11Conv()
        model = model.to(args.device)
    elif args.model == 'DenseNetFNN':
        model = DenseNetFNN(args.drop_out)
        model = model.to(args.device)
    elif args.model == 'ConvCrop':
        model = ConvCrop(args.drop_out)
        model = model.to(args.device)
    elif args.model == 'FNNCrop':
        model = FNNCrop()
        model = model.to(args.device)
    elif args.model == 'ConvFNNCrop':
        model = ConvFNNCrop()
        model = model.to(args.device)
    elif args.model == 'BayesianDenseNet':
        model = DenseNet(args.drop_out)
        model = BayesNet(args, model).to(args.device)
    elif args.model == 'BayesianDenseNetFNN':
        model = DenseNetFNN(args.drop_out)
        model = BayesNet(args, model).to(args.device)
    elif args.model == 'BayesianFNNCrop':
        model = FNNCrop()
        model = BayesNet(args, model).to(args.device)
    else:
        raise TypeError('Model is not defined')
    return model


def getOpt(args, model):
    criterion = torch.nn.MSELoss(reduction='sum')
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lrs == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lrs == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.995)
    elif args.lrs == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

    logger = {}
    logger['rmse_train'] = []
    logger['rmse_test'] = []
    logger['r2_train'] = []
    logger['r2_test'] = []

    return criterion, optimizer, scheduler, logger


