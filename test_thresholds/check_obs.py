import h5py
import mpi4py.MPI as MPI
import os
import numpy as np
import configparser
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for r in range(size):
    if rank==r:
        config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
        config = configparser.ConfigParser()
        config.read(config_home_path+'/test_config.txt')

        PTA      = config['current PTA']['PTA']
        PTA_path = config['PTAs'][PTA]

        #read hdf5 file to store observables
        obs_filepaths = '{}/{}/'.format(PTA_path, config['supporting directories']['obs'])
        obs_files     = [obs_filepaths + x for x in os.listdir(obs_filepaths)]
        obs_f = obs_files[r]

        Database_filepaths = '{}/{}/'.format(PTA_path, config['supporting directories']['Database'])
        Database_files     = [Database_filepaths + x for x in os.listdir(Database_filepaths)]
        Database_f = Database_files[r]

        high_ref_samples = []
        time_stamps_high_ref = []
        # for obs_f in obs_files:

        with h5py.File(obs_f, 'r') as hf_observables,\
             h5py.File(Database_f, 'r') as hf_database:
            obs_keys = list(hf_observables.keys())
            database_keys = list(hf_database.keys())
            print(obs_keys)
            print(database_keys)
            for time_stamp in obs_keys:
                cirrus_Ref = hf_observables[time_stamp + '/cirrus'][()]
                vis_Ref    = hf_observables[time_stamp + '/visRef'][()]
                cloud_Mask = hf_database[time_stamp + '/cloud_mask/Unobstructed_FOV_Quality_Flag']

                high_cirrus_obs_idx = np.where((cirrus_Ref > 0.4) & (cloud_Mask != 0))
                cirrus_Ref_          = cirrus_Ref[high_cirrus_obs_idx]

                num_pix_high        = high_cirrus_obs_idx[0].shape[0]
                if num_pix_high > 0:
                    # high_ref_samples.append(cirrus_Ref_)
                    # time_stamps_high_ref.append(time_stamp)
                    # print(high_ref_samples)
                    print(time_stamp, num_pix_high)
                # if time_stamp == '2010192.1825':
                #     # plt.imshow(cirrus_Ref, cmap='bone')
                #     plt.imshow(vis_Ref, cmap='bone')
                #     plt.colorbar()
                #     # plt.title(time_stamp+'1.38 microns BRF')
                #     plt.title(time_stamp+'0.65 microns BRF')
                #     plt.show()
