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
import operator
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from model.torchsummary import summary


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, drop_out):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.drop_out = drop_out
        if self.drop_out:
            self.drop1 = nn.Dropout2d(p=drop_out)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_out:
            out = self.drop1(out)
        return torch.cat([x, out], 1)


class EncodingBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, drop_out):
        super(EncodingBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.drop_out = drop_out
        if self.drop_out:
            self.drop1 = nn.Dropout2d(p=drop_out)
            self.drop2 = nn.Dropout2d(p=drop_out)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        if self.drop_out:
            out = self.drop1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_out:
            out = self.drop2(out)
        return out

    
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_out):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop_out = drop_out
        if self.drop_out:
            self.drop1 = nn.Dropout2d(p=drop_out)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        if self.drop_out:
            out = self.drop1(out)
        return out    

    
class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest', recompute_scale_factor=True)


class DecodingBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, scale_factor, drop_out):
        super(DecodingBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.upsample = UpsamplingNearest2d(scale_factor)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_out = drop_out
        if self.drop_out:
            self.drop1 = nn.Dropout2d(p=drop_out)
            self.drop2 = nn.Dropout2d(p=drop_out)
              
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        if self.drop_out:
            out = self.drop1(out)
        out = self.conv2(self.upsample(self.relu2(self.bn2(out))))
        if self.drop_out:
            out = self.drop2(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, kernel_size, stride, padding, drop_out):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, kernel_size, stride, padding, drop_out)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, kernel_size, stride, padding, drop_out):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, kernel_size, stride, padding, drop_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class LastLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LastLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.conv1(self.relu(self.bn1(x)))


class DenseNet(nn.Module):
    def __init__(self, drop_out):
        super(DenseNet, self).__init__()        
        self.conv_init = nn.Conv2d(36, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.block1 = DenseBlock(3, 128, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.encoder1 = EncodingBlock(176, 88, 88, drop_out=drop_out)
        self.block2 = DenseBlock(6, 88, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.decoder1 = DecodingBlock(184, 92, 92, 2., drop_out=drop_out)
        self.block3 = DenseBlock(3, 92, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.decoder2 = DecodingBlock(140, 70, 35, scale_factor=(70/36, 125/64), drop_out=drop_out)
        self.conv_last = LastLayer(35, 1)
        self._count_params()

    def forward(self, x):
        out = self.encoder1(self.block1(self.conv_init(x)))
        out = self.decoder2(self.block3(self.decoder1(self.block2(out))))
        out = self.conv_last(out)
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')
        
    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
                        
    
class DenseNet11Conv(nn.Module):
    def __init__(self):
        super(DenseNet11Conv, self).__init__()        
        self.conv_init = nn.Conv2d(36, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1 = DenseBlock(3, 128, 16, BasicBlock, kernel_size=1, stride=1, padding=0)
        self.transition1 = TransitionBlock(176, 88)
        self.block2 = DenseBlock(6, 88, 16, BasicBlock, kernel_size=1, stride=1, padding=0)
        self.transition2 = TransitionBlock(184, 92)
        self.block3 = DenseBlock(3, 92, 16, BasicBlock, kernel_size=1, stride=1, padding=0)
        self.transition3 = TransitionBlock(140, 70)
        self.transition4 = TransitionBlock(70, 35)
        self.conv_last = nn.Conv2d(35, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self._count_params()

    def forward(self, x):
        out = self.transition1(self.block1(self.conv_init(x)))
        out = self.transition4(self.transition3(self.block3(self.transition2(self.block2(out)))))
        out = self.conv_last(out)
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        
        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
                        

class DenseNetFNN(nn.Module):
    def __init__(self, drop_out):
        super(DenseNetFNN, self).__init__()        
        self.conv_init = nn.Conv2d(36, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.block1 = DenseBlock(3, 128, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.encoder1 = EncodingBlock(176, 88, 88, drop_out=drop_out)
        self.conv_linear1 = nn.Conv2d(88, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop1 = nn.Dropout2d(p=drop_out)
        self.conv_linear2 = nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop2 = nn.Dropout2d(p=drop_out)
        self.linear = nn.Linear(576, 576, bias=False)
        self.drop3 = nn.Dropout(p=drop_out) 
        self.block2 = DenseBlock(3, 48, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.decoder1 = DecodingBlock(184, 92, 92, 2., drop_out=drop_out)
        self.block3 = DenseBlock(3, 92, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.decoder2 = DecodingBlock(140, 70, 35, scale_factor=(70/36, 125/64), drop_out=drop_out)
        self.conv_last = LastLayer(35, 1)
        self._count_params()

    def forward(self, x):
        out = self.encoder1(self.block1(self.conv_init(x)))
        out_linear = self.conv_linear1(out)
        out_linear = self.drop1(out_linear)
        shapeInfor = out_linear.shape
        out_linear = out_linear.view(shapeInfor[0], -1)
        out_linear = self.linear(out_linear)
        out_linear = self.drop3(out_linear)
        out_linear = out_linear.view(shapeInfor[0], shapeInfor[1], shapeInfor[2], shapeInfor[3])
        out_linear = self.conv_linear2(out_linear)
        out_linear = self.drop2(out_linear)
        out_linear = self.block2(out_linear)
        out_conv = out
        out = torch.cat([out_conv, out_linear], 1)
        out = self.decoder2(self.block3(self.decoder1(out)))
        out = self.conv_last(out)
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        
        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')
        
    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


class ConvCrop(nn.Module):
    def __init__(self, drop_out):
        super(ConvCrop, self).__init__()        
        self.conv_init = nn.Conv2d(36, 80, kernel_size=7, stride=2, padding=3, bias=False)
        self.block1 = DenseBlock(3, 80, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.encoder1 = EncodingBlock(128, 64, 64, drop_out=drop_out)
        self.block2 = DenseBlock(3, 64, 16, BasicBlock, kernel_size=3, stride=1, padding=1, drop_out=drop_out)
        self.encoder2 = EncodingBlock(112, 56, 56, drop_out=drop_out)
        self.conv_last = nn.Conv2d(56, 1, kernel_size=3, stride=2, padding=[2,1], bias=False)
        self._count_params()

    def forward(self, x):
        out = self.conv_last(self.encoder2(self.block2(self.encoder1(self.block1(self.conv_init(x))))))
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        
        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
                        

class FNNCrop(nn.Module):
    def __init__(self):
        super(FNNCrop, self).__init__()        
        self.layer1 = nn.Conv2d(36, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer2 = EncodingBlock(32, 16, 16)
        self.layer3 = EncodingBlock(16, 8, 8)
                
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(576)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(576, 192, bias=False)

        self.bn3 = nn.BatchNorm1d(192)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(192, 48, bias=False)
        
        self._count_params()

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(x)))
        out = self.conv1(self.relu1(self.bn1(out)))
        
        shapeInfor = out.shape
        out = out.view(shapeInfor[0], -1)
        out = self.linear2(self.relu2(self.bn2(out)))
        out = self.linear3(self.relu3(self.bn3(out)))
        out = out.view(shapeInfor[0], 6, 8)        
        
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        
        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
                        
                        
class ConvFNNCrop(nn.Module):
    def __init__(self):
        super(ConvFNNCrop, self).__init__()        
        self.layer1 = nn.Conv2d(36, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer2 = EncodingBlock(32, 16, 16)
        self.layer3 = EncodingBlock(16, 8, 8)
                
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(576)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(576, 576, bias=False)

        self.bn3 = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn4 = nn.BatchNorm2d(24)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self._count_params()

    def forward(self, x):
        xCrop = x[:, :, 52:58, 52:60]

        out = self.layer3(self.layer2(self.layer1(x)))
        out = self.conv1(self.relu1(self.bn1(out)))
        
        shapeInfor = out.shape
        out = out.view(shapeInfor[0], -1)
        out = self.linear2(self.relu2(self.bn2(out)))

        out = out.view(shapeInfor[0], 12, 6, 8)        
        out = torch.cat((xCrop, out), 1)

        out = self.conv3(self.relu3(self.bn3(out)))
        out = self.conv4(self.relu4(self.bn4(out)))
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        
        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
  
                        
if __name__ == '__main__':
    x = torch.Tensor(10, 36, 70, 125)
    model = ConvFNNCrop()
    summary(model, (36, 70, 125), device='cpu')


    