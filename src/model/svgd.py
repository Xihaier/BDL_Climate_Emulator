'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import math
import torch
import numpy as np
import torch.nn as nn

from time import time
from utils.misc import r2Score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils.BayesTools import log_sum_exp, parameters_to_vector, vector_to_parameters


class SVGD(object):
    def __init__(self, args, bayes_nn, train_loader):
        self.bayes_nn = bayes_nn
        self.train_loader = train_loader
        self.n_samples = args.nSVGD
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.lr = args.lr
        self.lr_noise = args.lr_noise
        self.ntrain = args.ntrain
        self.batch_train_size = args.btrain
        self.out_channels = 1
        self.weight_decay = args.weight_decay
        self.lrs = args.lrs
        self.model = args.model
        self.device = args.device
        self.log_freq = args.log_freq
        self.log_print = args.log_print
        self.optimizers, self.schedulers = self._optimizers_schedulers(self.lr, self.lr_noise)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def _squared_dist(self, X):
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)

    def _Kxx_dxKxx(self, X):
        squared_dist = self._squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square
        return Kxx, dxKxx
    
    def _optimizers_schedulers(self, lr, lr_noise):
        optimizers = []
        schedulers = []
        
        for i in range(self.n_samples):
            if self.model == 'BayesianDenseNet':
                parameters = [{'params': [self.bayes_nn[i].log_beta], 'lr': lr_noise}, 
                              {'params': self.bayes_nn[i].conv_init.parameters()}, 
                              {'params': self.bayes_nn[i].block1.parameters()}, 
                              {'params': self.bayes_nn[i].encoder1.parameters()}, 
                              {'params': self.bayes_nn[i].block2.parameters()}, 
                              {'params': self.bayes_nn[i].decoder1.parameters()}, 
                              {'params': self.bayes_nn[i].block3.parameters()}, 
                              {'params': self.bayes_nn[i].decoder2.parameters()}, 
                              {'params': self.bayes_nn[i].conv_last.parameters()}]
            elif self.model == 'BayesianDenseNetFNN':
                parameters = [{'params': [self.bayes_nn[i].log_beta], 'lr': lr_noise}, 
                              {'params': self.bayes_nn[i].conv_init.parameters()}, 
                              {'params': self.bayes_nn[i].block1.parameters()}, 
                              {'params': self.bayes_nn[i].encoder1.parameters()}, 
                              {'params': self.bayes_nn[i].conv_linear1.parameters()}, 
                              {'params': self.bayes_nn[i].conv_linear2.parameters()}, 
                              {'params': self.bayes_nn[i].linear.parameters()}, 
                              {'params': self.bayes_nn[i].block2.parameters()}, 
                              {'params': self.bayes_nn[i].decoder1.parameters()}, 
                              {'params': self.bayes_nn[i].block3.parameters()}, 
                              {'params': self.bayes_nn[i].decoder2.parameters()}, 
                              {'params': self.bayes_nn[i].conv_last.parameters()}]
            elif self.model == 'BayesianFNNCrop':
                parameters = [{'params': [self.bayes_nn[i].log_beta], 'lr': lr_noise}, 
                              {'params': self.bayes_nn[i].layer1.parameters()}, 
                              {'params': self.bayes_nn[i].layer2.parameters()}, 
                              {'params': self.bayes_nn[i].layer3.parameters()}, 
                              {'params': self.bayes_nn[i].bn1.parameters()}, 
                              {'params': self.bayes_nn[i].relu1.parameters()}, 
                              {'params': self.bayes_nn[i].conv1.parameters()}, 
                              {'params': self.bayes_nn[i].bn2.parameters()}, 
                              {'params': self.bayes_nn[i].relu2.parameters()}, 
                              {'params': self.bayes_nn[i].linear2.parameters()}, 
                              {'params': self.bayes_nn[i].bn3.parameters()},
                              {'params': self.bayes_nn[i].relu3.parameters()},
                              {'params': self.bayes_nn[i].linear3.parameters()}]
            else:
                TypeError('Model is not defined')
                               
            optimizer_i = torch.optim.AdamW(parameters, lr=lr, weight_decay=self.weight_decay)
            optimizers.append(optimizer_i)

            if self.lrs == 'StepLR':
                schedulers.append(StepLR(optimizer_i, step_size=self.step_size, gamma=self.gamma))
            elif self.lrs == 'ReduceLROnPlateau':
                schedulers.append(ReduceLROnPlateau(optimizer_i, mode='min', factor=0.1, patience=10, verbose=True))
            else:
                raise TypeError('Scheduler is not defined')
                    
        return optimizers, schedulers

    def train(self, epoch, logger):
        print('-'*37)
        print('Training epoch {} summary'.format(epoch))
        
        self.bayes_nn.train()
        start = time()
        mse_train, r2_score = 0., 0.
        for batch_idx, (input_x, true_y) in enumerate(self.train_loader):
            input_x, true_y = input_x.to(self.device), true_y.to(self.device)
            self.bayes_nn.zero_grad()
            pred_y = torch.zeros_like(true_y)
            grad_log_joint = []
            theta = []
            log_joint = 0.

            for idx in range(self.n_samples):
                pred_y_i = self.bayes_nn[idx].forward(input_x)
                pred_y_i = torch.squeeze(pred_y_i)
                pred_y += pred_y_i.detach()
                log_joint_i = self.bayes_nn._log_joint(idx, pred_y_i, true_y, self.ntrain)
                log_joint_i.backward()
                log_joint += log_joint_i.item()

                vec_param, vec_grad_log_joint = parameters_to_vector(self.bayes_nn[idx].parameters(), both=True)
                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                theta.append(vec_param.unsqueeze(0))

            theta = torch.cat(theta)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint)
            grad_logp = torch.mm(Kxx, grad_log_joint)

            ######
            # new experiments: set dxKxx = 0 as suggested by Balu.
            # grad_theta = - (grad_logp + dxKxx)/self.n_samples
            grad_theta = - (grad_logp)/self.n_samples
            ######

            for idx in range(self.n_samples):
                vector_to_parameters(grad_theta[idx], self.bayes_nn[idx].parameters(), grad=True)
                self.optimizers[idx].step()
            
            mse_train += self.criterion(pred_y/self.n_samples, true_y).item()            
            r2_score += r2Score(pred_y/self.n_samples, true_y)
            if not (batch_idx%self.log_print):
                print('Batch {}, r2_score: {:.6f}'.format(batch_idx, r2_score/(batch_idx+1)))

        rmse_train = np.sqrt(mse_train/((batch_idx+1)*true_y.shape[0]))
        r2_train = r2_score/(batch_idx+1)
        for idx in range(self.n_samples):
            if self.lrs == 'ReduceLROnPlateau':
                self.schedulers[idx].step(rmse_train)
            else:
                self.schedulers[idx].step()
        stop = time()  

        print('Training epoch {} loss: {:.6f}  time: {:.3f}'.format(epoch, rmse_train, stop-start))
        print('-'*37+'\n')

        if epoch % self.log_freq == 0:
            logger['r2_train'].append(r2_train)
            logger['rmse_train'].append(rmse_train)


