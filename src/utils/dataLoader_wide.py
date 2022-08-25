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
import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader


class ClimateDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, data_crop):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_crop = data_crop

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data[index : index+self.seq_len, :, :]
        if self.data_crop:
            y = self.data[index+self.seq_len-1+self.pred_len, 52:58, 52:60]
        else:
            y = self.data[index+self.seq_len-1+self.pred_len, :, :]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def getDataloaders(args):
    data_dir = args.data_dir
    n_train = args.ntrain
    n_test = args.ntest
    batch_train_size = args.btrain
    batch_test_size = args.btest
    seq_len = args.seq_len
    pred_len = args.pred_len
    data_mask = args.data_mask
    data_crop = args.data_crop
    
    data_original = np.load(data_dir)
    mu = np.mean(data_original, axis=0)
    data = data_original - mu

    idx_sep = n_train+seq_len-1+pred_len
    idx_end = idx_sep+n_test+seq_len-1+pred_len

    data = data[:idx_end]
    
    if data_mask:  
        mask = copy.deepcopy(data[0,:,:])
        for idx1 in range(mask.shape[0]):
            for idx2 in range(mask.shape[1]):
                if idx1>=52 and idx1<=57 and idx2>=52 and idx2<=59:
                    mask[idx1][idx2] = 1
                else:
                    mask[idx1][idx2] = 0
        
        for idx in range(data.shape[0]):
            data[idx,:,:] = np.multiply(mask, data[idx,:,:])
    
    data_flatten = np.reshape(data, (data.size, 1))
#     data_map = MinMaxScaler(feature_range=(-1, 1))
    data_map = StandardScaler()
    data_map.fit(data_flatten)
    data_flatten_minmax = data_map.transform(data_flatten)
    data_minmax = np.reshape(data_flatten_minmax, (data.shape[0], data.shape[1], data.shape[2]))

    train_data = data_minmax[:idx_sep]
    test_data = data_minmax[idx_sep:idx_end]

    train_dataset = ClimateDataset(train_data, seq_len=seq_len, pred_len=pred_len, data_crop=data_crop)
    test_dataset = ClimateDataset(test_data, seq_len=seq_len, pred_len=pred_len, data_crop=data_crop)

    train_loader = DataLoader(train_dataset, batch_size=batch_train_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_test_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    mapDic = {}
    mapDic['y'] = data_map

    return train_loader, test_loader, mapDic


class ClimateTrainSet(Dataset):
    def __init__(self, data, seq_len, pred_len, data_crop):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_crop = data_crop

    def __len__(self):
        return 1280

    def __getitem__(self, index):
        idx = index*11
        x = self.data[idx : idx+self.seq_len, :, :]
        if self.data_crop:
            y = self.data[idx+self.seq_len-1+self.pred_len, 52:58, 52:60]
        else:
            y = self.data[idx+self.seq_len-1+self.pred_len, :, :]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class ClimateTestSet(Dataset):
    def __init__(self, data, seq_len, pred_len, data_crop):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_crop = data_crop

    def __len__(self):
        return 128

    def __getitem__(self, index):
        idx = index*11
        x = self.data[idx : idx+self.seq_len, :, :]
        if self.data_crop:
            y = self.data[idx+self.seq_len-1+self.pred_len, 52:58, 52:60]
        else:
            y = self.data[idx+self.seq_len-1+self.pred_len, :, :]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def getWideDataloaders(args):
    data_dir = args.data_dir
    n_train = args.ntrain
    n_test = args.ntest
    batch_train_size = args.btrain
    batch_test_size = args.btest
    seq_len = args.seq_len
    pred_len = args.pred_len
    data_mask = args.data_mask
    data_crop = args.data_crop
    
    data_original = np.load(data_dir)
    mu = np.mean(data_original, axis=0)
    data = data_original - mu
    
    if data_mask:  
        mask = copy.deepcopy(data[0,:,:])
        for idx1 in range(mask.shape[0]):
            for idx2 in range(mask.shape[1]):
                if idx1>=52 and idx1<=57 and idx2>=52 and idx2<=59:
                    mask[idx1][idx2] = 1
                else:
                    mask[idx1][idx2] = 0
        
        for idx in range(data.shape[0]):
            data[idx,:,:] = np.multiply(mask, data[idx,:,:])
    
    data_flatten = np.reshape(data, (data.size, 1))
#     data_map = MinMaxScaler(feature_range=(-1, 1))
    data_map = StandardScaler()
    data_map.fit(data_flatten)
    data_flatten_minmax = data_map.transform(data_flatten)
    data_minmax = np.reshape(data_flatten_minmax, (data.shape[0], data.shape[1], data.shape[2]))

    idx_sep = 11*(1280-1)+48
    train_data = data_minmax[:idx_sep]
    idx_sep = 11*(128-1)+48
    test_data = data_minmax[5:(5+idx_sep)]
    
    train_dataset = ClimateTrainSet(train_data, seq_len=seq_len, pred_len=pred_len, data_crop=data_crop)
    test_dataset = ClimateTestSet(test_data, seq_len=seq_len, pred_len=pred_len, data_crop=data_crop)

    train_loader = DataLoader(train_dataset, batch_size=batch_train_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_test_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    mapDic = {}
    mapDic['y'] = data_map

    return train_loader, test_loader, mapDic



