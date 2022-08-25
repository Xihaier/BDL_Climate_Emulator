import torch
import numpy as np

from time import time
from config import Parser
from model.nn import getModel, getOpt
from utils.dataLoader import getDataloaders
from utils.misc import r2Score, dataTranform, saveCNN, postProcessing

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
def train(epoch, logger):
    print('-'*37)
    print('Training epoch {} summary'.format(epoch))

    model.train()
    start = time()
    mse, r2_score = 0., 0.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        model.zero_grad()
        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        mse += loss.item()
        r2_score += r2Score(outputs, targets)
        print('Batch {}, r2_score: {:.6f}'.format(batch_idx, r2_score/(batch_idx+1)))

    rmse = np.sqrt(mse/(outputs.shape[0]*(batch_idx+1)))
    r2_score = r2_score/(batch_idx+1)
    if args.lrs == 'ReduceLROnPlateau':
        scheduler.step(rmse)
    else:
        scheduler.step()
    stop = time()

    print('Training epoch {} loss: {:.6f}  time: {:.3f}'.format(epoch, rmse, stop-start))
    print('-'*37+'\n')

    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_score)
        logger['rmse_train'].append(rmse)

#####################################
# step 4: define test
#####################################
def test(epoch, logger):
    print('-'*37)
    print('Test epoch {} summary'.format(epoch))

    model.eval()
    start = time()
    mse, r2_score = 0., 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        mse += loss.item()
        r2_score += r2Score(outputs, targets)
        print('Batch {}, r2_score: {:.6f}'.format(batch_idx, r2_score/(batch_idx+1)))

        if epoch >= args.save_epoch:
            if batch_idx == 0: 
                out_save = dataTranform(outputs, dataMap)
                tar_save = dataTranform(targets, dataMap)
            else:
                out_save = np.concatenate((out_save, dataTranform(outputs, dataMap)))
                tar_save = np.concatenate((tar_save, dataTranform(targets, dataMap)))

    rmse = np.sqrt(mse/(outputs.shape[0]*(batch_idx+1)))
    r2_score = r2_score/(batch_idx + 1)
    stop = time()
    
    print('Test epoch {} loss: {:.6f}  time: {:.3f}'.format(epoch, rmse, stop-start))
    print('-'*37+'\n')

    if (epoch >= args.save_epoch) and (rmse<=min(logger['rmse_test'])):
        saveCNN(out_save, tar_save, epoch, args.save_dir)
    
    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_score)
        logger['rmse_test'].append(rmse)

#####################################
# step 5: start training and test
#####################################
tic = time()
for epoch in range(1, args.epochs + 1):
    train(epoch, logger)
    with torch.no_grad():
        test(epoch, logger)
tic2 = time()
print('-'*37)
print('Training and test summary')
print('Finished training {} epochs with {} data using {} seconds' .format(args.epochs, args.ntrain, tic2 - tic))
print('-'*37+'\n')

#####################################
# step 6: summarize results
#####################################
del model, train_loader, test_loader
postProcessing(logger, args)
