'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def r2Score(output, targets):
    output = toNumpy(output)
    targets = toNumpy(targets)
    r2 = 0.
    for idx in range(output.shape[0]):
        r2 += r2_score(output[idx,:,:], targets[idx,:,:])
    return r2/output.shape[0]


def saveCNN(outputs, targets, epoch, save_dir):
    np.savez(os.path.join(save_dir, 'results'), outputs=outputs, targets=targets, epoch=epoch)


def saveBCNN(outputs, targets, uncer, y_ensemble, epoch, save_dir):
    np.savez(os.path.join(save_dir, 'results'), targets=targets, outputs=outputs, uncertainties=uncer, outputsEnsemble=y_ensemble, epoch=epoch)


def dataEnsembleTranform(tensor, data_map):
    data = toNumpy(tensor)
    for idxBatch in range(data.shape[1]):
        dataBatch = data[:,idxBatch,:,:]
        for idxParticle in range(data.shape[0]):
            if idxParticle == 0:
                temp = np.expand_dims(data_map.inverse_transform(dataBatch[idxParticle,:,:]), axis=0)
            else:
                temp = np.concatenate((temp, np.expand_dims(data_map.inverse_transform(dataBatch[idxParticle,:,:]), axis=0)))
        if idxBatch == 0:
            out = np.expand_dims(temp, axis=0)
        else:
            out = np.concatenate((out, np.expand_dims(temp, axis=0)), axis=0)
    return out
    

def postProcessing(logger, args):
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 27})
    
    x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
    r2_train = logger['r2_train']
    r2_test = logger['r2_test']

    plt.figure(figsize=(15, 12))
    plt.plot(x_axis, r2_train, 'k-', label='training')
    plt.plot(x_axis, r2_test, 'r--', label='test')
    plt.xlabel('Epoch')
    plt.ylabel(r'$R^2$-score')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.save_dir + '/r2.png', bbox_inches='tight')
    plt.close()

    rmse_train = logger['rmse_train']
    rmse_test = logger['rmse_test']

    plt.figure(figsize=(15, 12))
    plt.plot(x_axis, rmse_train, 'k-', label='training')
    plt.plot(x_axis, rmse_test, 'r--', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.save_dir + '/rmse.png', bbox_inches='tight')
    plt.close()

    np.savetxt(args.save_dir + '/r2_train.txt', r2_train)
    np.savetxt(args.save_dir + '/r2_test.txt', r2_test)
    np.savetxt(args.save_dir + '/rmse_train.txt', rmse_train)
    np.savetxt(args.save_dir + '/rmse_test.txt', rmse_test)


def dataTranform(tensor, data_map):
    data = toNumpy(tensor)
    for idx in range(data.shape[0]):
        if idx == 0:
            temp = np.expand_dims(data_map.inverse_transform(data[idx,:,:]), axis=0)
        else:
            temp = np.concatenate((temp, np.expand_dims(data_map.inverse_transform(data[idx,:,:]), axis=0)))
    return temp


