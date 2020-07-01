def accuracy_by_sunViewGeometry(conf_matx_path, DOY_bin, result_path):

    with h5py.File(conf_matx_path, 'r') as hf_confmatx:
        bins = np.array(list(hf_confmatx.keys()))

        #polar plot 2 dims; [r,theta] -> [VZA, RAZ]; accuarcy at each [i,j]

        #2 accuracies -> [combined, cloud, clear, f1 score]
        accuracy = np.zeros((len(bins), 3))
        num_samples = np.zeros((len(bins)))

        num_VZA_bins    = 15
        num_RAZ_bins    = 13
        num_cosSZA_bins = 10
        correct_sum = np.zeros((num_cosSZA_bins, num_VZA_bins, num_RAZ_bins))
        total_sum    = np.zeros((num_cosSZA_bins, num_VZA_bins, num_RAZ_bins))
        for i, bin_ID in enumerate(bins):
            conf_matx = hf_confmatx[bin_ID][()]
            num_samples[i] = conf_matx.sum()

            #[cosSZA, VZA, RAZ, sceneID]
            OLP = [int(bin_ID[24:26]), int(bin_ID[31:33]), \
                   int(bin_ID[38:40]), int(bin_ID[55:57])  ]

            #so for each bin ID, sum true pos and true negative
            #and sum the total
            #then save into correct_sum & toal_sum
            correct_sum[OLP[0], OLP[1], OLP[2]] += np.nansum(conf_matx[:2])
            total_sum[OLP[0], OLP[1], OLP[2]]   += np.nansum(conf_matx)
            #print(bin_ID)

        total_sum[total_sum==0]=np.nan
        accuracy_total = correct_sum / total_sum

        # return accuracy_total*100

        #write MCM_accuracy and num_samples to disk
        np.savez('{}/bin_accuracy_DOY_bin_{:02d}.npz'.format(result_path, DOY_bin),\
                  MCM_accuracy=accuracy_total)#, num_samples=num_samples)

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
            accuracy_by_sunViewGeometry(conf_matx_path, DOY_bin, result_path)
