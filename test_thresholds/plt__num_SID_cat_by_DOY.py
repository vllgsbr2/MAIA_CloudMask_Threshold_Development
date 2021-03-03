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

# import mpi4py.MPI as MPI
import configparser

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
#
# for r in range(size):
#     if rank==r:
#         DOY_bin = r

#count SGW and SI over all data from output file
config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

#get output files
filepath_output = PTA_path + '/' + config['supporting directories']['MCM_Output']+'/numKmeansSID_16'
timestamps      = os.listdir(filepath_output)


# DOY_sfcID = np.zeros((20,46,18))
# for yr in range(2002,2020):
#     for r in range(46):
#         DOY_bin = r
#         DOY_end         = (DOY_bin+1)*8
#         DOY_start       = DOY_end - 7
#
#         timestamps_      = [x for x in timestamps if (DOY_start<int(x[4:7])<=DOY_end and int(x[:4])==yr)]
#         filepath_output_ = [filepath_output+'/'+x+'/MCM_Output.h5' for x in timestamps_]
#
#         for l, f in enumerate(filepath_output_):
#             with h5py.File(f, 'r') as hf_output:
#                 SID = hf_output['Ancillary/scene_type_identifier'][()]
#                 idx, x=np.unique(SID[SID>=0], return_counts=True)
#                 idx=idx.astype(np.int)
#             DOY_sfcID[idx ,DOY_bin, yr-2002] += x
#             print(timestamps_[l])
#
# np.savez('/data/keeling/a/vllgsbr2/c/DOY_sfcID_yr_by_yr.npz'.format(DOY_bin), DOY_sfcID=DOY_sfcID)

data = np.load('/data/keeling/a/vllgsbr2/c/DOY_sfcID_yr_by_yr.npz')
dataset_names = data.files

DOY_sfcID = data[dataset_names[0]]
import matplotlib.colors as colors
# f, ax = plt.subplots(nrows=6, ncols=3)
#
# for i, a in enumerate(ax.flat):
#     im=a.imshow(DOY_sfcID[:,:,i], cmap='jet', vmin=0, vmax=DOY_sfcID.max())#, norm=colors.LogNorm(vmin=1, vmax=10**7))
#     if i>=15:
#         a.set_xticks(np.arange(46))
#         a.set_xticklabels(np.arange(8,376,8), rotation=45)
#     else:
#         a.set_xticks([])
#     if i==0 or i==3 or i==6 or i==9 or i==12 or i==15:
#         a.set_yticks(np.arange(20))
#         a.set_yticklabels(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','Coast','Water','Sun-Glint','Snow'], rotation=45)
#     else:
#         a.set_yticks([])
#     a.set_ylabel(2002+i)
#
# cax = f.add_axes([0.92, 0.23, 0.01, 0.5])#l,b,w,h
# cbar = f.colorbar(im, cax=cax)

DOY_sfcID_single = np.sum(DOY_sfcID, axis=2)
DOY_sfcID_single_snowglint = DOY_sfcID_single[-2:,:]
DOY_sfcID_single_dark = np.sum(DOY_sfcID_single[:11,:], axis=0)
DOY_sfcID_single_light = np.sum(DOY_sfcID_single[11:,:], axis=0)
light_SID_percent_over_time = 100*DOY_sfcID_single_light/(DOY_sfcID_single_dark+DOY_sfcID_single_light)
# for i in range()
DOY = np.arange(46)
f,ax = plt.subplots(nrows=1,ncols=1)
plt.rcParams['font.size'] = 18
ax.plot(DOY, DOY_sfcID_single_snowglint[0,:], label='sun-glint', linewidth=3)
ax.plot(DOY, DOY_sfcID_single_snowglint[1,:], label='snow-ice', linewidth=3)
ax_twin = ax.twinx()
ax_twin.plot(DOY, light_SID_percent_over_time, label='% SID 11-15', color='g', linewidth=3)
ax.set_xticks(DOY)
ax.set_xticklabels(np.arange(8,376,8), rotation=45)
ax_twin.set_ylabel('[%]')
ax.set_ylabel('raw count')
ax.set_xlabel('DOY bins [Julian Calendar]')
bad_DOYs = [24,48,88,120,160,232,288,344]
for i in bad_DOYs:
    ax.axvline((i/8)-1, linestyle='dashed', color="grey")
ax.legend()
ax_twin.legend()

plt.show()
