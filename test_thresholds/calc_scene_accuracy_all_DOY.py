
def scene_conf_matx(conf_matx_path, DOY_bin, result_path):

    with h5py.File(conf_matx_path, 'r') as hf_confmatx:
        confmatx_keys = np.array(list(hf_confmatx.keys()))
        time_stamps   = [x[-12:] for x in confmatx_keys]
        masks         = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_mask']
        tables        = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_table']
        #print(masks)
        #pixel by pixel accuracy
        accuracy = np.zeros((1000,1000,4))
        #number of samples that contributed to every evaluation type
        #needed since not all pixels in each scene have data, -999 instead
        num_samples = np.zeros((1000,1000,4))

        for i, mask in enumerate(masks):
            print(mask[-12:])
            mask = hf_confmatx[mask][()]
            present_data_idx = np.where((mask != -999) & (mask != 0))
            no_data_idx    = np.where(mask == -999)
            #set mask to zero for no data so it doesnt contribute to accuracy
            mask[no_data_idx] = 0
            #add count to num samples where data is present
            num_samples[present_data_idx] += 1

            # xx = np.shape(present_data_idx)[1]/1e6
            # print(time_stamps[i], xx)

            accuracy += mask

        #print(accuracy[:,:,0])
        #% of time mask is correct at each location, when data is present
        MCM_accuracy = np.sum(accuracy[:,:,:2], axis=2) / np.sum(num_samples, axis=2)#num_samples[:,:,0]

        #write MCM_accuracy and num_samples to disk
        np.savez('{}/scene_Accuracy_DOY_bin_{:02d}.npz'.format(result_path, DOY_bin),\
                  MCM_accuracy=MCM_accuracy, num_samples=num_samples)

        # return MCM_accuracy

if __name__ == '__main__':

    import h5py
    import os
    import numpy as np
    import tables
    tables.file._open_files.close_all()
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database'
            DOY_bin = r
            conf_matx_path = '{}/conf_matx_scene_all_DOY/conf_matx_scene_DOY_bin_{:02d}.HDF5'.format(home, DOY_bin)
            result_path = '{}/{}'.format(home, 'scene_accuracy')
            scene_conf_matx(conf_matx_path, DOY_bin, result_path)

















            #
