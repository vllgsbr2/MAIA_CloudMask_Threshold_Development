import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import mpi4py.MPI as MPI
import os
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for r in range(size):
    if rank==r:
            data_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/MCM_Output/'
            time_stamps = os.listdir(data_home)
            MCM_Output_paths = [data_home + x + '/MCM_Output.HDF5' for x in time_stamps]

            #assign subset of files to current rank
            end               = len(MCM_Output_paths)
            processes_per_cpu = end // (size-1)
            start             = rank * processes_per_cpu

            if rank < (size-1):
                end = (rank+1) * processes_per_cpu
            elif rank==(size-1):
                processes_per_cpu_last = end % (size-1)
                end = (rank * processes_per_cpu) + processes_per_cpu_last

            MCM_Output_paths = MCM_Output_paths[start:end]
            time_stamps = time_stamps[start:end]
            #open files
            for output_path, time_stamp in zip(MCM_Output_paths, time_stamps):
                with h5py.File(output_path, 'r') as hf_MCM_output:

                    final_cloud_mask = hf_MCM_output['cloud_mask_output/final_cloud_mask'][()]
                    BRF_band_4       = hf_MCM_output['Reflectance/band_04'][()]
                    BRF_band_5       = hf_MCM_output['Reflectance/band_05'][()]
                    BRF_band_6       = hf_MCM_output['Reflectance/band_06'][()]
                    observable_data  = hf_MCM_output['cloud_mask_output/observable_data'][()]
                    DTT              = hf_MCM_output['cloud_mask_output/DTT'][()]

                    #reformat observable_data/DTT for plotting
                    WI, NDVI, NDSI, VIS_Ref, NIR_Ref, SVI, Cirrus =\
                                                        observable_data[:,:,0],\
                                                        observable_data[:,:,1],\
                                                        observable_data[:,:,2],\
                                                        observable_data[:,:,3],\
                                                        observable_data[:,:,4],\
                                                        observable_data[:,:,5],\
                                                        observable_data[:,:,6]

                    DTT_WI, DTT_NDVI, DTT_NDSI, DTT_VIS_Ref, DTT_NIR_Ref,\
                    DTT_SVI, DTT_Cirrus =   DTT[:,:,0],\
                                            DTT[:,:,1],\
                                            DTT[:,:,2],\
                                            DTT[:,:,3],\
                                            DTT[:,:,4],\
                                            DTT[:,:,5],\
                                            DTT[:,:,6]

                #plotting*******************************************************************
                cmap = 'bwr'

                vmin = -1.2
                vmax = 1.2
                l,w, = 20,8

                #final cloud mask
                #f1 = plt.figure(figsize=(20,10))
                f1, ax1 = plt.subplots(ncols=2, figsize=(l,w), sharex=True, sharey=True)

                ax1[0].imshow(final_cloud_mask, cmap='Greys')
                #ax1[0].set_title('final MAIA CLoud Mask')

                ax1[0].set_xticks([])
                ax1[0].set_yticks([])

                from rgb_enhancement import *

                RGB = np.dstack((BRF_band_6, BRF_band_5, BRF_band_4))

                #RGB = get_enhanced_RGB(RGB)
                ax1[1].imshow(RGB)
                ax1[1].set_xticks([])
                ax1[1].set_yticks([])


                #observables
                f0, ax0 = plt.subplots(ncols=4, nrows=2, figsize=(l,w), sharex=True, sharey=True)
                im = ax0[0,0].imshow(WI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax0[0,1].imshow(NDVI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax0[0,2].imshow(NDSI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax0[0,3].imshow(VIS_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
                ax0[1,0].imshow(NIR_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
                ax0[1,1].imshow(SVI*2, cmap=cmap, vmin=vmin, vmax=vmax)
                ax0[1,2].imshow(Cirrus, cmap=cmap, vmin=vmin, vmax=vmax)

                ax0[1,3].imshow(RGB)

                ax0[0,0].set_title('WI')
                ax0[0,1].set_title('NDVI')
                ax0[0,2].set_title('NDSI')
                ax0[0,3].set_title('VIS_Ref')
                ax0[1,0].set_title('NIR_Ref')
                ax0[1,1].set_title('SVI x2 scaling')
                ax0[1,2].set_title('Cirrus')
                ax0[1,3].set_title('BRF RGB')


                cb_ax = f0.add_axes([0.93, 0.1, 0.02, 0.8])
                cbar = f0.colorbar(im, cax=cb_ax)

                for a in ax0.flat:
                    a.set_xticks([])
                    a.set_yticks([])

                #DTT
                vmin = -101
                vmax = 101

                cmap = cm.get_cmap('bwr')
                #cmap.set_bad(color='black')
                cmap.set_under('black')

                f, ax = plt.subplots(ncols=4, nrows=2, figsize=(l,w), sharex=True, sharey=True)
                im = ax[0,0].imshow(DTT_WI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[0,1].imshow(DTT_NDVI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[0,2].imshow(DTT_NDSI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[0,3].imshow(DTT_VIS_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[1,0].imshow(DTT_NIR_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[1,1].imshow(DTT_SVI, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[1,2].imshow(DTT_Cirrus, cmap=cmap, vmin=vmin, vmax=vmax)

                ax[1,3].imshow(RGB)

                ax[0,0].set_title('DTT_WI')
                ax[0,1].set_title('DTT_NDVI')
                ax[0,2].set_title('DTT_NDSI')
                ax[0,3].set_title('DTT_VIS_Ref')
                ax[1,0].set_title('DTT_NIR_Ref')
                ax[1,1].set_title('DTT_SVI')
                ax[1,2].set_title('DTT_Cirrus')
                ax[1,3].set_title('BRF RGB')

                cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
                cbar = f.colorbar(im, cax=cb_ax)

                for a in ax.flat:
                    a.set_xticks([])
                    a.set_yticks([])

                #create directory to story plots in if it doesn't exist
                directory_name = '{}/{}'.format(data_home, time_stamp)
                if not(os.path.exists(directory_name)):
                    os.mkdir(directory_name)

                f1.savefig('{}/{}_cloud_mask.png'.format(directory_name, time_stamp), dpi=300)
                f.savefig('{}/{}_DTT.png'.format(directory_name, time_stamp), dpi=300)
                f0.savefig('{}/{}_obs.png'.format(directory_name, time_stamp), dpi=300)
                plt.close(fig=f)
                plt.close(fig=f1)
                plt.close(fig=f0)
                print('{} plots saved'.format(time_stamp))
