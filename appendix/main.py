from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import argparse
from os import listdir
from os.path import join
from convLSTM import ConvLSTM
from convDenseLSTM import ConvDenseLSTM
from LSTM import LSTM
from MLP import MLP
import numpy as np
from sklearn import decomposition

parser = argparse.ArgumentParser(description='PyTorch UNet')
parser.add_argument('--pred',type=int,default=1, help='prediction length')
parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train_dir',type=str,default='train',help='directory for training data')
parser.add_argument('--val_dir',type=str,default='val',help='directory for validation data')
parser.add_argument('--test_dir',type=str,default='test',help='directory for testing data')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--order', type=int, default=36)
parser.add_argument('--hidden', nargs='+', type=int, default = [512,256],help='hidden dimension')
parser.add_argument('--metric',type=str,default='MSE',help='choose the type of error metric: MSE, (non-dimensional)RMSE')
parser.add_argument('--model', '-m', type=str, default='convLSTM', help='choose which model is going to use')
parser.add_argument('--nGPU', '-n', type=int, default=1, help='choose # of GPUs')
parser.add_argument('--save', '-s', type=str,default=True,help='a folder for saving' )
parser.add_argument('--load', '-l', type=str,default=True,help='a folder for lodaing' )
parser.add_argument('--num_class', '-nc', type=int, default=8, help='set # of classes')
parser.add_argument('--num_layer', '-nl', type=int, default=2, help='set # of layers for LSTM')


args = parser.parse_args()

pca = decomposition.PCA(n_components=20)

print(args.hidden)

class dataset(Dataset):
    def __init__(self,dat, type,pred_len, input_transform=None,target_transform=None):
        print('data loading')
        print(dat.shape)


        tempData = dat #dat['temp']
        # self.lons = dat['lons'] #longitude
        temp_len = tempData.shape[0]
        train_len = temp_len // 10 * 6
        inter = temp_len // 10 * 2

        if type == 'train':
            #self.temp = np.load('train_data.npy')
            self.temp = tempData[:train_len,:,:] #temperature
            np.save('train_data.npy',self.temp)
            if(args.model == 'PCALSTM'):
                data = self.temp.reshape((self.temp.shape[0],-1))
                self.tempX = pca.fit_transform(data)
        elif type == 'val':
            #self.temp = np.load('val_data.npy')
            self.temp = tempData[train_len:train_len + inter,:,:]
            np.save('val_data.npy', self.temp)
            if (args.model == 'PCALSTM'):
                data = self.temp.reshape((self.temp.shape[0], -1))
                self.tempX = pca.transform(data)
        elif type == 'test':
            #self.temp = np.load('test_data.npy')
            self.temp = tempData[train_len+inter:,:,:]
            np.save('test_data.npy', self.temp)
            if (args.model == 'PCALSTM'):
                data = self.temp.reshape((self.temp.shape[0], -1))
                self.tempX = pca.transform(data)


        # self.lats = dat['lats'] #latitude
        self.pred_len = pred_len

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.order = args.order
        print(type, self.temp.shape, self.temp.shape[0] - self.order * 2 + 1 - self.pred_len)


    def __len__(self):
        return self.temp.shape[0] - self.order - self.pred_len + 1 #self.temp.shape[0]//self.order - self.pred_len

    def __getitem__(self, index):
        # print(index)
        if args.model == 'PCALSTM':
            # x = self.tempX[index * self.order:(index + 1) * self.order, :]
            # y = self.tempX[(index + 1) * self.order + self.pred_len, :]
            # z = self.temp[(index+1) * self.order + self.pred_len,:,:]
            x = self.tempX[index : index + self.order, :]
            y = self.tempX[index + self.order - 1 + self.pred_len, :]
            z = self.temp[index + self.order -1 + self.pred_len,:,:]

            return x, y,z
        else:
            # x = self.temp[index * self.order:(index + 1) * self.order,:,:]
            # y = self.temp[(index+1) * self.order + self.pred_len,:,:]
            x = self.temp[index : index + self.order, :, :]
            y = self.temp[index + self.order - 1 + self.pred_len,:,:]

            return x,y

def error(pred,target):
    return pred - target

def meanx(dat):
    return np.mean(dat,axis=(0,1))

def rmss(dat):
    return np.sqrt((dat ** 2).mean(axis=0))

def rmst(dat):
    return np.sqrt((dat ** 2).mean(axis=0))

#train, validation, and test a model
def train(optimizer, scheduler, num_epochs):
    mseloss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        out_array = []
        pred_array = []
        target_array = []

        # Each epoch has a training and validation phase
        datasets = ['train','val']
        if epoch == num_epochs - 1:
            datasets = ['train','val','test']

        for phase in datasets:
            since = time.time()
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_samples = 0
            total_loss = 0
            idx = 0
            if args.model == 'PCALSTM':
                for inputs, pca_targets, targets in dataloaders[phase]:
                    inputs = inputs.type(torch.FloatTensor)
                    pca_targets = pca_targets.type(torch.FloatTensor)
                    inputs = inputs.unsqueeze(2)
                    inputs = inputs.to(device)
                    pca_targets = pca_targets.to(device)

                    # zero the parameter gradients
                    if phase == 'train':
                        optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    outputs = outputs[:, -1, -1, :, :].squeeze(1).squeeze(2)
                    loss = mseloss(outputs, pca_targets)
                    tem = outputs.detach().cpu().numpy()
                    invPCA = torch.Tensor(pca.inverse_transform(tem).reshape((-1,192,288)))
                    targets = targets.type(torch.FloatTensor)

                    if phase == 'test':
                        err = invPCA - targets
                        if idx == 0:
                            pred_array = invPCA
                            target_array = targets
                            out_array = err
                        else:
                            pred_array = np.concatenate((pred_array,invPCA))
                            target_array = np.concatenate((target_array,targets))
                            out_array = np.concatenate((out_array,err))

                    total_loss += mseloss(invPCA,targets)
                    # backward + optimize only if in training phase
                    # if not args.model == 'MLP':
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # compute # of samples
                    epoch_samples += inputs.size(0)
                    idx +=1
            elif args.model == 'PM':
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.type(torch.FloatTensor)
                    targets = targets.type(torch.FloatTensor)
                    inputs = inputs.unsqueeze(2)
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # zero the parameter gradients
                    if phase == 'train':
                        optimizer.zero_grad()

                    outputs = inputs[:,35,0,:,:]
                    loss = mseloss(inputs[:, 35, 0, :, :], targets)

                    if phase == 'test':
                        err = outputs.detach().cpu().numpy() - targets.detach().cpu().numpy()
                        if idx == 0:
                            pred_array = outputs.detach().cpu().numpy()
                            target_array = targets.detach().cpu().numpy()
                            out_array = err
                        else:
                            pred_array = np.concatenate((pred_array,outputs.detach().cpu().numpy()))
                            target_array = np.concatenate((target_array,targets.detach().cpu().numpy()))
                            out_array = np.concatenate((out_array,err))
                    total_loss += loss

                    #compute # of samples
                    epoch_samples += inputs.size(0)
                    idx += 1
            else:
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.type(torch.FloatTensor)
                    targets = targets.type(torch.FloatTensor)
                    inputs = inputs.unsqueeze(2)
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # zero the parameter gradients
                    if phase == 'train':
                        optimizer.zero_grad()

                    # forward
                    # track history if only in train

                    with torch.set_grad_enabled(phase == 'train'):
                        if args.model == 'convLSTM' or args.model == 'convDenseLSTM':
                            outputs,_ = model(inputs)
                            outputs = outputs[0]
                        else:
                            outputs = model(inputs)

                    loss = mseloss(outputs[:,-1:,-1,:,:].squeeze(1), targets)

                    if phase == 'test':
                        err = outputs[:,-1:,-1,:,:].squeeze(1).detach().cpu().numpy() - targets.detach().cpu().numpy()
                        if idx == 0:
                            pred_array = outputs[:,-1:,-1,:,:].squeeze(1).detach().cpu().numpy()
                            target_array = targets.detach().cpu().numpy()
                            out_array = err
                        else:
                            pred_array = np.concatenate((pred_array,outputs[:,-1:,-1,:,:].squeeze(1).detach().cpu().numpy()))
                            target_array = np.concatenate((target_array,targets.detach().cpu().numpy()))
                            out_array = np.concatenate((out_array,err))
                    total_loss += loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    #compute # of samples
                    epoch_samples += inputs.size(0)
                    idx += 1
            time_elapsed = time.time() - since
            print('Epoch {}/{} - {}:MSE:{} time:{:.0f}m {:.000f}s'.format(epoch, num_epochs - 1, phase,
                                                                        total_loss / epoch_samples, time_elapsed // 60,
                                                                        time_elapsed % 60))
    final_loss = meanx(rmss(out_array))/meanx(rmst(test_set.temp))
    np.save('error'+args.model+str(args.pred)+'_t_'+str(args.order)+'.npy',out_array)
    np.save('pred_'+args.model+str(args.pred)+'_t_'+str(args.order)+'.npy',pred_array)
    np.save('target_'+args.model+str(args.pred)+'_t_'+str(args.order)+'.npy',target_array)
    np.save('pca'+args.model+str(args.pred)+'_t_'+str(args.order)+'.npy',pca)
    print('Epoch {}/{} - {}:RMSE:{} time:{:.0f}m {:.0f}s'.format(epoch, num_epochs - 1,phase,final_loss,time_elapsed // 60, time_elapsed % 60))


# set training, validation, testing datasets
filename = '../SST.npy'#'dat/climate.npz'#'dat/test_dat.npz'#'dat/climate.npz'
dat = np.load(filename)
dat = dat[:2400,:,:]

train_set = dataset(dat, 'train',args.pred)
val_set = dataset(dat,'val',args.pred)
test_set = dataset(dat,'test',args.pred)


dataloaders = {
    'train': DataLoader(train_set, batch_size=args.batchSize, shuffle=False, num_workers=0),
    'val': DataLoader(val_set, batch_size=args.batchSize, shuffle=False, num_workers=0),
    'test': DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, num_workers=0)
}

#set a device
device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(device)

#set a model
if args.model == 'convLSTM':
    model = ConvLSTM(input_size=(70, 125),
                 input_dim=1,
                 hidden_dim=[8, 8],
                 kernel_size=(5, 5),
                 num_layers=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
elif args.model == 'convDenseLSTM':
    model = ConvDenseLSTM(input_size=(70, 125),  #(192,288)
                 input_dim=1,
                 hidden_dim=[8, 8],
                 kernel_size=(5, 5),
                 num_layers=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
elif args.model == 'MLP':
    model = MLP(input_size=(192,288),time_seq=args.order,hidden_dim=(args.hidden[0] ,args.hidden[1]),dropout=args.dropout)
elif args.model == 'LSTM':
    model = LSTM(input_size=(192,288),time_seq=args.order,hidden_dim=3000,num_layers=args.num_layer)
elif args.model == 'PCALSTM':
    model = LSTM(input_size=(20,1),time_seq=args.order,hidden_dim=256,pca=True)
else:
    model = LSTM(input_size=(192, 288), time_seq=args.order, hidden_dim=3000, num_layers=args.num_layer)
# model.load_state_dict(torch.load(args.model+ str(args.pred) +'.pth'))

#set a optimizer (Adam)
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model = model.to(device)
train(optimizer_ft, exp_lr_scheduler, num_epochs=args.epoch) #train a model
torch.save(model.state_dict(), args.model + str(args.pred)+str(args.order) +'.pth') #save a trained model

