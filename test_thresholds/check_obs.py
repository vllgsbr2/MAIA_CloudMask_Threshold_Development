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

        input_home = '{}/{}'.format(PTA_path, config['supporting directories']['MCM_Input'])

        # Database_filepaths = '{}/{}/'.format(PTA_path, config['supporting directories']['Database'])
        # Database_files     = [Database_filepaths + x for x in os.listdir(Database_filepaths)]
        # Database_f = Database_files[r]


        high_ref_samples = []
        time_stamps_high_ref = []
        # for obs_f in obs_files:

        with h5py.File(obs_f, 'r') as hf_observables:
            obs_keys = list(hf_observables.keys())

            for time_stamp in obs_keys:
                Database_f = '{}/test_JPL_data_{}.h5'.format(input_home, time_stamp)
                with h5py.File(Database_f, 'r') as hf_database:
                    cirrus_Ref = hf_observables[time_stamp + '/cirrus'][()]
                    vis_Ref    = hf_observables[time_stamp + '/visRef'][()]
                    cloud_Mask = hf_database['MOD35_cloud_mask'][()]

                    high_cirrus_obs_idx = np.where((cirrus_Ref > 0.4) & (cloud_Mask != 0))
                    cirrus_Ref_          = cirrus_Ref[high_cirrus_obs_idx]

                    num_pix_high        = high_cirrus_obs_idx[0].shape[0]
                    if num_pix_high > 0:
                        # high_ref_samples.append(cirrus_Ref_)
                        # time_stamps_high_ref.append(time_stamp)
                        # print(high_ref_samples)
                        print(time_stamp, num_pix_high)
                    if time_stamp == '2010197.1845':
                        print(high_cirrus_obs_idx)
                        # plt.imshow(cirrus_Ref, cmap='jet', vmin=0, vmax = 0.4)
                        # plt.title(time_stamp+' 1.38 microns BRF')
                        # plt.title(time_stamp+' 0.65 microns BRF')
                        # plt.imshow(vis_Ref, cmap='bone')
                        # plt.colorbar()
                        # plt.show()

                        f, ax = plt.subplots(ncols=2)
                        ax[0].imshow(cirrus_Ref, cmap='jet', vmin=0, vmax = 0.4)
                        ax[0].set_title(time_stamp+' 1.38 microns BRF')

                        ax[1].imshow(cloud_Mask, cmap='binary')
                        ax[1].set_title(time_stamp+' cloud mask')

                        plt.show()
