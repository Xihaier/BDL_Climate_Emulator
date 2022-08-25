import sys
from os import cpu_count

# import dask
# import dask.array as da
# from dask import delayed, compute

from numpy import *
from matplotlib.pyplot import *
ion()
from scipy.stats.stats import pearsonr

import pickle
#import cdms2, cdutil
#from cdms2 import MV2
from eofs.cdms import Eof

import cartopy.crs as ccrs



nchunks = 1
dsamp = 1 #max(1, 100//nchunks)
num_workers = cpu_count() - 4

print(cpu_count(), ' cores; dsamp is %d; nchunks is %d'%(dsamp, nchunks))
#from dask.distributed import Client
#client = Client()

#print('Dashboard initialized')

dat = np.load('dat/climate.npz')
lons = dat['lons']
lats = dat['lats']
# eof_file = '/home/balu/tas_prediction/pre_post/data/eof12.pkl'
# neofs = 20

# with open(eof_file, 'rb') as f:
#     eofobj = pickle.load(f)

#eofs = eofobj.eofs(neofs=neofs)
# lons, lats = eofs[0].getLongitude()[:], eofs[0].getLatitude()[:]
# #wts = eofobj.getWeights()

# #prefix = '11_' #This was used for the CI2019 submission
# #prefix = '16_'
# prefix = '1_' #Attention model of 6Aug2019

# #dir = '/Users/balu/Downloads/Changlin/' #CI2019 submission data is on
# Mac laptop
# #dir = '/home/balu/Downloads/Changlin/6Aug2019/'
# dir = '/home/balu/tas_prediction/exp/'
case = 'synE'

# prefix = dir+case+'/test/1_'

#inputs = load(prefix + 'inputs.npy', allow_pickle=True)
#outputs = load(dir+prefix + 'outputs.npy', allow_pickle=True)
#targets = load(dir+prefix + 'targets.npy', allow_pickle=True)
import numpy as np
errs = np.load('pred_convLSTM1.npy')#[::dsamp]
refs = np.load('target_convLSTM1.npy')#[::dsamp]
shp = errs.shape
print(shp)
# print(refs.shape)
errs -= refs #uses less memory

if nchunks>1:
    errs = da.from_array(errs, chunks=(shp[0]//nchunks, shp[1], shp[2]))
    refs = da.from_array(refs, chunks=(shp[0]//nchunks, shp[1], shp[2]))
    '''
    err_fld = delayed(tensordot)(errs, eofs.data, 1)
    ref_fld = delayed(tensordot)(refs, eofs.data, 1)
    err_fld *= err_fld
    ref_fld *= ref_fld
    rmse_fld = delayed(sqrt)(delayed(mean)(err_fld, axis=0) /
delayed(mean)(ref_fld, axis=0))
    '''

    err_tmp = da.tensordot(errs, eofs.data, 1)
    err_tmp *= err_tmp
    err_fld = err_tmp.mean(axis=0)

    ref_tmp = da.tensordot(refs, eofs.data, 1)
    ref_tmp *= ref_tmp
    ref_fld = ref_tmp.mean(axis=0)

    print("Computional Graph Setup")
    err_fld, ref_fld = compute(err_fld, ref_fld)
    err2d = sqrt(err_fld / ref_fld)
    print("DONE computing!")

else:
    err_fld = errs#ensordot(errs, eofs.data, 1)
    ref_fld = refs#tensordot(refs, eofs.data, 1)
    err_fld *= err_fld
    ref_fld *= ref_fld
    err2d = sqrt(mean(err_fld, axis=0) / mean(ref_fld, axis=0))

print(err2d.shape)


levels=linspace(0.4,1.0,21)


fgnm = case + '_%dm'%(0+1)
fig = figure(fgnm); clf()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
cntr = ax.contourf(lons, lats, err2d, levels=levels,
            cmap=cm.RdBu_r, transform=ccrs.PlateCarree())
colorbar(cntr, orientation='horizontal', pad=0.1)
tight_layout()
savefig(fgnm, bbox_inches='tight')

ilon, ilat = 32, 32
fgnm = case + '_%d-%d'%(ilon,ilat)
figure(fgnm); clf()
plot(err2d[:, ilon, ilat])
savefig(fgnm)

# sys.exit(1)

# err_nd = sqrt(mean(errs*errs, axis=0) / mean(vrfs*vrfs, axis=0))
#
# shp = prds.shape
# acc = zeros(shp[1:])
# for t in range(shp[1]):
#     for m in range(shp[2]):
#         acc[t, m] = pearsonr(prds[:,t,m], vrfs[:,t,m])[0]
#
#
#
# fgnm = case + 'acc_lead'
# figure(fgnm)
# clf()
# [plot(acc[:36, i], alpha=max(0.1, (neofs-i)/neofs)) for i in range(neofs)]
# xlabel('Prediction Lead Time (months)')
# ylabel('Anomaly Correlation')
# tight_layout()
# savefig(fgnm)
#
#
# fgnm = case + 'rmse_lead'
# figure(fgnm)
# clf();
# [plot(err_nd[:36, i], alpha=max(0.1, (neofs-i)/neofs)) for i in
# range(neofs)]
# xlabel('Prediction Lead Time (months)')
# ylabel('Non-dimensional RMS Error')
# tight_layout()
# savefig(fgnm)

