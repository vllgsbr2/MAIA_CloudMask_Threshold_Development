import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import configparser
import os
import h5py

# config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
# config = configparser.ConfigParser()
# config.read(config_home_path+'/test_config.txt')
#
# PTA          = config['current PTA']['PTA']
# PTA_path     = config['PTAs'][PTA]

# #get sfc IDs
# filepath_SID = PTA_path + '/' + config['supporting directories']['Surface_IDs']
# filepath_SID = [filepath_SID +'/'+ x for x in os.listdir(filepath_SID) if x[0]=='s']
# SID = np.zeros((400,300,46))
# DOY_sfcID = np.zeros((18,46))
# for i in range(46):
#     with Dataset(filepath_SID[i],'r') as nc_sfcID:
#         SID[:,:,i] = nc_sfcID.variables['surface_ID'][:,:]
#         idx, x=np.unique(SID[:,:,i], return_counts=True)
#         idx=idx.astype(np.int)
#     DOY_sfcID[idx,i] = x
#
# plt.imshow(DOY_sfcID, cmap='jet')
# plt.colorbar()
# plt.xticks(np.arange(46), np.arange(8,376,8))
# plt.yticks(np.arange(18))
# plt.show()

import mpi4py.MPI as MPI
import configparser

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
#
# for r in range(size):
#     if rank==r:
#         DOY_bin = r
DOY_bin = 0
#count SGW and SI over all data from output file
config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

#get output files
filepath_output = PTA_path + '/' + config['supporting directories']['MCM_Output']+'/numKmeansSID_16'
timestamps      = os.listdir(filepath_output)
DOY_end         = DOY_bin*8
DOY_start       = DOY_end - 7
timestamps      = [x fox in timestamps if DOY_start<int(x[4:7])<=DOY_end]
filepath_output = [filepath_output+'/'+x+'/MCM_Output.h5' for x in timestamps]

DOY_sfcID = np.zeros(20)
for f in filepath_output:
    with h5py.File(f, 'r') as hf_output:
        SID = hf_output['Ancillary/scene_type_identifier'][()]
        idx, x=np.unique(SID[SID>=0], return_counts=True)
        idx=idx.astype(np.int)
    DOY_sfcID[idx] += x

np.savez('/data/keeling/a/vllgsbr2/c/DOY_sfcID_{:03d}.npz'.format(DOY_bin), DOY_sfcID=DOY_sfcID)
