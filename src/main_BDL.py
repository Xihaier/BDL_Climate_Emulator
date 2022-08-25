import torch
import numpy as np

from time import time
from config import Parser
from model.svgd import SVGD
from model.nn import getModel, getOpt
from utils.dataLoader import getDataloaders
from utils.misc import toNumpy, r2Score, dataTranform, dataEnsembleTranform, saveBCNN, postProcessing

#####################################
# step 1: load data
#####################################
args = Parser().parse()
train_loader, test_loader, mapDic = getDataloaders(args)
dataMap = mapDic['y']

#####################################
# step 2: define the model
#####################################
model = getModel(args)
criterion, optimizer, scheduler, logger = getOpt(args, model)

#####################################
# step 3: define training
#####################################
model_SVGD = SVGD(args, model, train_loader)

#####################################
# step 4: define test
#####################################
def test(epoch, logger):
    print('-'*37)
    print('Test epoch {} summary'.format(epoch))

    model.eval()
    start = time()
    mse_test, r2_score = 0., 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        mse, _, outputs = model._compute_mse_nlp(inputs, targets, size_average=True, out=True)
        outputs_mean = outputs.mean(0)
        EyyT = (outputs ** 2).mean(0)
        EyEyT = outputs_mean ** 2
        outputs_noise_var = (- model.log_beta).exp().mean()
        outputs_var =  EyyT - EyEyT + outputs_noise_var
        mse_test += mse
        r2_score += r2Score(torch.squeeze(outputs_mean), targets)
        print('Batch {}, r2_score: {:.6f}'.format(batch_idx, r2_score/(batch_idx+1)))
        
        if epoch >= args.save_epoch:
            if batch_idx == 0: 
                out_save = dataTranform(torch.squeeze(outputs_mean), dataMap)
                tar_save = dataTranform(targets, dataMap)
                uncer_save = np.sqrt(toNumpy(torch.squeeze(outputs_var))) * 2
                outEnsemble_save = dataEnsembleTranform(torch.squeeze(outputs), dataMap)
            else:
                out_save = np.concatenate((out_save, dataTranform(torch.squeeze(outputs_mean), dataMap)))
                tar_save = np.concatenate((tar_save, dataTranform(targets, dataMap)))
                uncer_save = np.concatenate((uncer_save, np.sqrt(toNumpy(torch.squeeze(outputs_var))) * 2))
                outEnsemble_save = np.concatenate((outEnsemble_save, dataEnsembleTranform(torch.squeeze(outputs), dataMap)))      

    rmse_test = np.sqrt(mse_test/(outputs_mean.shape[0]*(batch_idx+1)))
    r2_score = r2_score/(batch_idx+1)
    stop = time()
    
    print('Test epoch {} loss: {:.6f}  time: {:.3f}'.format(epoch, rmse_test, stop-start))
    print('-'*37+'\n')

    if (epoch >= args.save_epoch) and (rmse_test<=min(logger['rmse_test'])):
        saveBCNN(out_save, tar_save, uncer_save, outEnsemble_save, epoch, args.save_dir)    
    
    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_score)
        logger['rmse_test'].append(rmse_test)

#####################################
# step 5: start training and test
#####################################
tic = time()
for epoch in range(1, args.epochs + 1):
    model_SVGD.train(epoch, logger)
    with torch.no_grad():
        test(epoch, logger)
tic2 = time()
print('-'*37)
print('Training and test summary')
print('Finished training {} epochs with {} data and {} SVGD samples using {} seconds' .format(args.epochs, args.ntrain, args.nSVGD, tic2 - tic))
print('-'*37+'\n')

#####################################
# step 4: summarize results
#####################################
del model, train_loader, test_loader
postProcessing(logger, args)
