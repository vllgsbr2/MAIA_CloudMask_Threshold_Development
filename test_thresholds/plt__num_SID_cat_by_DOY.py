import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import configparser
import os

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

#get sfc IDs
filepath_SID = PTA_path + '/' + config['supporting directories']['Surface_IDs']
filepath_SID = [filepath_SID +'/'+ x for x in os.listdir(filepath_SID) if x[0]=='s']
SID = np.zeros((400,300,46))
DOY_sfcID = np.zeros((18,46))
for i in range(46):
    with Dataset(filepath_SID[i],'r') as nc_sfcID:
        SID[:,:,i] = nc_sfcID.variables['surface_ID'][:,:]
        idx, x=np.unique(SID[:,:,i], return_counts=True)
        idx=idx.astype(np.int)
    DOY_sfcID[idx,i] = x
print(DOY_sfcID.shape)
plt.imshow(DOY_sfcID, cmap='jet')
plt.colorbar()
plt.xticks(np.arange(46))
plt.xticklabels(np.arange(8,376,8))
plt.yticks(np.arange(18))
plt.show()
