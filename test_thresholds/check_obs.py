import h5py
import mpi4py.MPI as MPI
import os
import numpy as np
import configparser

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

        high_ref_samples = []
        time_stamps_high_ref = []
        # for obs_f in obs_files:
        obs_f = obs_files[r]
        with h5py.File(obs_f, 'r') as hf_observables:
            obs_keys = list(hf_observables.keys())
            for time_stamp in obs_keys:
                cirrus_Ref = hf_observables[time_stamp + '/cirrus'][()]

                high_cirrus_obs_idx = np.where(cirrus_Ref > 0.4)
                cirrus_Ref          = cirrus_Ref[high_cirrus_obs_idx]

                num_pix_high        = high_cirrus_obs_idx[0].shape[0]
                if num_pix_high > 0:
                    high_ref_samples.append(cirrus_Ref)
                    time_stamps_high_ref.append(time_stamp)
                    print(high_ref_samples)
                    print(time_stamp, num_pix_high)
