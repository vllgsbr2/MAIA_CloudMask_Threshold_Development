import h5py
import mpi4py.MPI as MPI
import os
import numpy as np
import configparser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

                        import matplotlib.colors as matCol
                        from matplotlib.colors import ListedColormap
                        cmap = ListedColormap(['white', 'green', 'blue','black'])
                        norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)

                        f, ax = plt.subplots(ncols=2)
                        im0 = ax[0].imshow(cirrus_Ref, cmap='jet', vmin=0, vmax = 0.4)
                        ax[0].set_title(time_stamp+' 1.38 microns BRF')
                        im0.cmap.set_under('black')

                        im1 = ax[1].imshow(cloud_Mask, cmap=cmap, norm=norm)
                        ax[1].set_title(time_stamp+' cloud mask')



                        divider0 = make_axes_locatable(ax[0])
                        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
                        f.colorbar(im0, cax=cax0, orientation='vertical')



                        divider1 = make_axes_locatable(ax[1])
                        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                        cbar1 = f.colorbar(im1, cax=cax1, orientation='vertical')

                        cbar1.set_ticks([0.5,1.5,2.5,3.5])
                        cbar1.set_ticklabels(['cloudy', 'uncertain\nclear', \
                                             'probably\nclear', 'confident\nclear'])

                        for a in ax:
                            a.set_yticks([])
                            a.set_xticks([])

                        plt.show()
