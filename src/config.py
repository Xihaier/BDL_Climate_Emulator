import os
import errno
import torch
import argparse

from pprint import pprint
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class DenseNetConfig:
    '''
    Configuration Dataclass to setup the model for the DenseNet.
    '''    
    # Data parameters
    btrain: int = 128
    btest: int = 128

        
@dataclass
class DenseNet11Config:
    '''
    Configuration Dataclass to setup the model for the DenseNet.
    '''     
    # Data parameters
    btrain: int = 64
    btest: int = 64
        

@dataclass
class DenseNetFNNConfig:
    '''
    Configuration Dataclass to setup the model for the DenseNetFNN.
    '''    
    # Data parameters
    btrain: int = 128
    btest: int = 128

        
@dataclass
class BayesianDenseNetConfig:
    '''
    Configuration Dataclass to setup the model for the Bayesian DenseNet.
    '''    
    # Data parameters
    btrain: int = 32
    btest: int = 32

    # Stein Variational Gradient Descent
    nSVGD: int =20
    w_prior_shape: float = 1.
    w_prior_rate: float = 0.05
    beta_prior_shape: float = 2.
    beta_prior_rate: float = 1.e-6
    lr_noise: float = 0.01
        
    # logging
    log_print: int = 4


@dataclass
class BayesianDenseNetFNNConfig:
    '''
    Configuration Dataclass to setup the model for the Bayesian DenseNetFNN.
    '''    
    # Data parameters
    btrain: int = 32
    btest: int = 32

    # Stein Variational Gradient Descent
    nSVGD: int =20
    w_prior_shape: float = 1.
    w_prior_rate: float = 0.05
    beta_prior_shape: float = 2.
    beta_prior_rate: float = 1.e-6
    lr_noise: float = 0.01
        
    # logging
    log_print: int = 4


Config_Mapping = OrderedDict(
    [
        ('DenseNet',  DenseNetConfig),
        ('DenseNet11',  DenseNet11Config),
        ('DenseNetFNN',  DenseNetFNNConfig),
        ('ConvCrop',  DenseNetConfig),
        ('FNNCrop',  DenseNetConfig),
        ('ConvFNNCrop',  DenseNetConfig),
        ('BayesianDenseNet', BayesianDenseNetConfig),
        ('BayesianDenseNetFNN', BayesianDenseNetFNNConfig),
    ]
)


class Parser(argparse.ArgumentParser):
    def __init__(self):
        '''
        Program arguments: note: Use `python main.py --help` for more information.
        '''
        super(Parser, self).__init__(description='Bayesian Deep Climate Emulator')
        # experiment
        self.add_argument('--model', type=str, default='DenseNet', choices=['DenseNet', 'DenseNet11', 'DenseNetFNN', 'ConvCrop', 'FNNCrop', 'ConvFNNCrop', 'BayesianDenseNet', 'BayesianDenseNetFNN'], help='choose the model')
        self.add_argument('--exp-dir', type=str, default="./results", help='directory to save experiments')
        self.add_argument('--seq-len',type=int, default=36, help='sequence length')
        self.add_argument('--pred-len',type=int, default=1, help='prediction length')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')

        # data
        self.add_argument('--data-dir', type=str, default='SST.npy', help='directory to load data')
        self.add_argument('--data-mask', type=bool, default=False, help='mask data or not')
        self.add_argument('--data-crop', type=bool, default=False, help='crop data or not')
        self.add_argument('--ntrain', type=int, default=1280, help='number of training data')
        self.add_argument('--ntest', type=int, default=128, help='number of test data')
        self.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
        self.add_argument('--pin_memory', type=bool, default=True, help='If True, the data loader will copy Tensors into CUDA pinned memory before returning them')

        # optimization
        self.add_argument('--lr', type=float, default=0.0007, help='ADAM learning rate')
        self.add_argument('--drop-out', type=float, default=0, help='dropout rate')
        self.add_argument('--gamma', type=float, default=0.99, help='ADAM learning rate')
        self.add_argument('--step-size', type=int, default=3, help='step size')
        self.add_argument('--lrs', type=str, default='StepLR', choices=['StepLR', 'ReduceLROnPlateau'], help="learning rate scheduler")
        self.add_argument('--weight-decay', type=float, default=1e-7, help="weight decay")

    def parse(self):
        '''
        Parse program arguments
        '''
        # Load basic configuration
        args = self.loadConfig(self.parse_args())
        
        # Set epochs
        epochs = {1: 700, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500, 7: 300, 8: 300, 9: 300, 10: 300, 11: 300, 12: 300, 13: 200, 14: 200, 15: 200, 16: 200, 17: 200, 18: 200}
        args.epochs = epochs[args.pred_len]

        saveEpochs = {key:2 for key in range(1, 19)}
        args.save_epoch = saveEpochs[args.pred_len]

        dropouts = {1: 0, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5, 17: 0.5, 18: 0.5}
        args.drop_out = dropouts[args.pred_len]

        # Experiment save directory
        args.save_dir = args.exp_dir + '/' + args.model + '/pred_{}'.format(args.pred_len)
        self.mkdirs(args.save_dir)

        # Set device
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Print arguments
        print('-'*37)
        print('Arguments summary')
        pprint(vars(args))
        print('-'*37+'\n')

        args.sepLineS = '-'*37
        args.sepLineE = '-'*37+'\n'

        return args

    def loadConfig(self, args):
        '''
        Loads experimental configurations.
        '''
        if(args.model in Config_Mapping.keys()):
            config_class = Config_Mapping[args.model]
            config = config_class()
            for attr, value in config.__dict__.items():
                if not hasattr(args, attr) or getattr(args, attr) is None:
                    setattr(args, attr, value)
        else:
            raise AssertionError("Provided experiment name, {:s}, not found in experiment list.".format(args.exp_type))

        return args

    def mkdirs(self, *directories):
        '''
        Makes a directory if it does not exist
        '''
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
