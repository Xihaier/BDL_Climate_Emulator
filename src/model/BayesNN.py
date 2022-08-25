'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import copy
import math
import torch
import torch.nn as nn

from torch.distributions import Gamma
from torch.nn.parameter import Parameter
from utils.BayesTools import log_sum_exp


class BayesNet(nn.Module):
    def __init__(self, args, model):
        super(BayesNet, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model {} is not a Module subclass".format(torch.typename(model)))

        self.n_samples = args.nSVGD
        self.device = args.device
        self.w_prior_shape = args.w_prior_shape
        self.w_prior_rate = args.w_prior_rate
        self.beta_prior_shape = args.beta_prior_shape
        self.beta_prior_rate = args.beta_prior_rate
        self.model = args.model
        self.criterion = torch.nn.MSELoss(reduction='sum')

        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        instances = []
        for i in range(self.n_samples):
            new_instance = copy.deepcopy(model)
            new_instance.reset_parameters()
            print('Reset parameters in model instance {}'.format(i))
            instances.append(new_instance)
        self.nnets = nn.ModuleList(instances)
        del instances

        log_beta = Gamma(self.beta_prior_shape, self.beta_prior_rate).sample((self.n_samples,)).log()
        for i in range(self.n_samples):
            self.nnets[i].log_beta = Parameter(log_beta[i])

        self._count_parameters()

    def __getitem__(self, idx):
        return self.nnets[idx]

    @property
    def log_beta(self):
        return torch.tensor([self.nnets[i].log_beta.item() for i in range(self.n_samples)], device=self.device)

    def forward(self, input):
        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(input))
        output = torch.stack(output)
        return output

    def _count_parameters(self):
        print('-'*37)
        print('Bayesian summary')  
        print('Total model parameters: %.2fM' % (sum(p.numel() for p in self.parameters())/1000000.0))
        print('Total model parameters: %.2fk' % (sum(p.numel() for p in self.parameters())/1000.0))
        print('-'*37+'\n')

    def _log_joint(self, index, output, target, ntrain):
        log_likelihood = ntrain / output.size(0) * (- 0.5 * self.nnets[index].log_beta.exp() * (target - output).pow(2).sum() + 0.5 * target.numel() * self.nnets[index].log_beta)
        log_prob_prior_w = torch.tensor(0.).to(self.device)

        if self.model == 'BayesianDenseNet':
            for param in self.nnets[index].conv_init.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].block1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].encoder1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].block2.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].decoder1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].block3.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].decoder2.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].conv_last.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        elif self.model == 'BayesianDenseNetFNN':
            for param in self.nnets[index].conv_init.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].block1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].encoder1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].conv_linear1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].conv_linear2.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].linear.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].block2.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].decoder1.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()            
            for param in self.nnets[index].block3.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].decoder2.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
            for param in self.nnets[index].conv_last.parameters():
                log_prob_prior_w += torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()                               
        else:
            TypeError('Model is not defined')   
                
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        log_prob_prior_log_beta = ((self.beta_prior_shape-1.0) * self.nnets[index].log_beta - self.nnets[index].log_beta.exp() * self.beta_prior_rate)        
        return log_likelihood + log_prob_prior_w + log_prob_prior_log_beta

    def _compute_mse_nlp(self, input, target, size_average=True, out=False):
        output = self.forward(input)
        log_beta = self.log_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        log_2pi_S = torch.tensor(0.5 * target[0].numel() * math.log(2 * math.pi) + math.log(self.n_samples), device=self.device)
        exponent = - 0.5 * (log_beta.exp() * ((target - output) ** 2)).view(self.n_samples, target.size(0), -1).sum(-1) + 0.5 * target[0].numel() * self.log_beta.unsqueeze(-1)
        nlp = - log_sum_exp(exponent, dim=0).mean() + log_2pi_S
        mse = self.criterion(torch.squeeze(output.mean(0)), torch.squeeze(target)).item()

        if not size_average:
            mse *= target.numel()
            nlp *= target.size(0)
        if not out:
            return mse, nlp
        else:
            return mse, nlp, output

    def predict(self, x_test):
        y = self.forward(x_test)
        y_pred_mean = y.mean(0)
        EyyT = (y ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        beta_inv = (- self.log_beta).exp()
        y_pred_var = beta_inv.mean() + EyyT - EyEyT
        return y_pred_mean, y_pred_var


