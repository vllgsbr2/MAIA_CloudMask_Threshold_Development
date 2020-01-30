def confusion_matrix(threshold_path):

    with h5py.File(threshold_path, 'r+') as hf_thresh:

        hf_keys    = list(hf_group.keys())
        num_points = len(hf_keys)

        #grab obs, CM, and thresholds
        cloud_mask = np.zeros((num_points))
        obs = np.empty((num_points, 7))

        for i, data_point in enumerate(hf_keys):
            dataset_path  = '{}/label_and_obs'.format(data_point)

            data = hf_group[dataset_path][()]
            cloud_mask[i] = int(data[0])
            obs[i,:]      = data[1:]


        thresholds = hf_thresh['thresholds'][()]

        #see if any obs trigger cloudy
        for i in range(num_points)
            cloudy_idx = np.where(obs[i,:] >= thresholds)
            clear_idx  = np.where(obs[i,:] <  thresholds)

        #compare with CM

        #both return cloudy
        true  = np.where(CM == thresholds[cloudy_idx]).sum()
        #both return clear
        false = np.where(CM == thresholds[clear_idx ]).sum()
        #MOD cloudy MAIA clear
        false_pos = np.where(CM != thresholds[cloudy_idx]).sum()
        #MOD clear MAIA cloudy
        false_neg = np.where(CM != thresholds[clear_idx ]).sum()

        #make result into confusion matrix
        conf_mat = np.array([true, false, false_pos, false_neg])

        #save it back into group/threshold file
        try:
            dataset = hf_thresh.create_dataset('confusion_matrix', data=conf_mat)
            dataset.attrs['label'] = ['both_cloudy, both_clear,\
                                       MOD_cloud_MAIA_clear, MOD_clear_MAIA_cloudy']
        except:
            hf_thresh['confusion_matrix'][:] = conf_mat

if __name__ == '__main__':

    import h5py
    import mpi4py.MPI as MPI
    import tables
    import os
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            #define paths for the three databases
            home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
            PTA_file_path    = home + 'group_DOY_05_60_cores/'
            database_files   = os.listdir(PTA_file_path)
            database_files   = [PTA_file_path + filename for filename in database_files]
            database_files   = np.sort(database_files)

            #define start and end file for a particular rank
            #(size - 1) so last processesor can take the modulus
            num_files        = len(database_files)
            files_per_cpu    = num_files//(size-1)
            start, end       = r*files_per_cpu, (r+1)*files_per_cpu

            if r==(size-1):
                start, end = r*files_per_cpu, r*files_per_cpu + num_files % size

            hf_group_paths = database_files[start:end]

            for path in hf_group_paths:
                confusion_matrix(path)
