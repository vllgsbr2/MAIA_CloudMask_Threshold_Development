import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys


def make_SID_MCM_rdy(home_og, home):
    '''
    home_og {str} -- path to original surface ID files for 1 PTA
    home {str} -- path to directory to put modified surface ID files for 1 PTA
    '''

    #grab OG surface IDs
    sfc_ID_paths = np.sort(np.array(os.listdir(home_og)))
    sfc_ID_paths = [x for x in sfc_ID_paths if x[:10] == 'surfaceID_']
    sfc_IDs = np.zeros((400,300,len(sfc_ID_paths)))

    for i, sfc_ID_path in enumerate(sfc_ID_paths):
        with Dataset(home_og + sfc_ID_path, 'r') as nc_sfc_ID:
            sfc_IDs[:,:,i] = nc_sfc_ID.variables['surface_ID'][:,:]
            # print(sfc_IDs[:,:,i].max())

    # 0 - Water
    # 1 - Coastline
    # 2 - Not-enough-sample
    # 3 - Darkest
    # .
    # .
    # .
    # 13- Brightest

    #modify the surface ID in memory
    sfc_IDs_mod = np.copy(sfc_IDs)
    #set sfcID 3 to 13 to 0 to 10
    sfc_IDs_mod[sfc_IDs >=3]  = sfc_IDs_mod[sfc_IDs >=3] - 3
    #set invalid to -999
    sfc_IDs_mod[sfc_IDs == 2] = -9
    #set coastline to 11
    sfc_IDs_mod[sfc_IDs == 1] = 11
    #set water (0) to 12
    sfc_IDs_mod[sfc_IDs == 0] = 12

    # im=plt.imshow(sfc_IDs_mod[:,:,0], cmap='jet', vmax=11)
    # im.cmap.set_over('k')
    # plt.show()
    # sys.exit()

    #copy files into new destination
    os.system('cp {}/surfaceID* {}/'.format(home_og, home))

    #edit copied surface IDs files with new surface ID
    sfc_ID_paths = np.sort(np.array(os.listdir(home)))
    sfc_ID_paths = [x for x in sfc_ID_paths if x[:10] == 'surfaceID_']

    for i, sfc_ID_path in enumerate(sfc_ID_paths):
        with Dataset(home + sfc_ID_path, 'r+') as nc_sfc_ID:
            nc_sfc_ID.variables['surface_ID'][:,:] = sfc_IDs_mod[:,:,i]

    sfc_IDs_read_mod = np.zeros((400,300,len(sfc_ID_paths)))
    for i, sfc_ID_path in enumerate(sfc_ID_paths):
        with Dataset(home + sfc_ID_path, 'r') as nc_sfc_ID:
            sfc_IDs_read_mod[:,:,i] = nc_sfc_ID.variables['surface_ID'][:,:]
            
    im=plt.imshow(sfc_IDs_read_mod[:,:,0], cmap='jet', vmin=0, vmax=11)
    im.cmap.set_over('k')
    im.cmap.set_under('pink')
    plt.show()
    sys.exit()


if __name__ == '__main__':
    import configparser

    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA      = config['current PTA']['PTA']
    # PTA_path = config['PTAs'][PTA]

    home        = '/data/gdi/c/gzhao1/MCM-surfaceID/SfcID/{}/'.format(PTA)
    destination = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/{}/Surface_IDs/'.format(PTA)
    make_SID_MCM_rdy(home, destination)
