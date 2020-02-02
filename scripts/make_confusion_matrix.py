def confusion_matrix(threshold_path):

    with h5py.File(threshold_path, 'r+') as hf_thresh:

        hf_keys    = list(hf_thresh.keys())
        hf_keys    = [x for x in hf_keys if x[:2] == 'ti']
        num_points = len(hf_keys)

        #grab obs, CM, and thresholds
        cloud_mask = np.zeros((num_points))
        obs = np.empty((num_points, 7))

        for i, data_point in enumerate(hf_keys):
            dataset_path  = '{}/label_and_obs'.format(data_point)

            data = hf_thresh[dataset_path][()]
            cloud_mask[i] = int(data[0])
            obs[i,:]      = data[1:]


        thresholds = hf_thresh['thresholds'][()]

        #calculate distance to threshold for NDxI 
        DTT_NDxI = np.zeros((num_points, 2)) 
        for i in range(1,3):
            NDxI          = obs[:,i]
            T             = thresholds[i]
            DTT_NDxI[:,i-1] = (T - np.abs(NDxI)) / T
    
        #see if any obs trigger cloudy
        #assume clear (i.e. 1), cloudy is 0
        cloud_mask_MAIA = np.ones((num_points))

        for i in range(num_points):
            for j in range(7):
                if j==23:
                    pass
                elif j>=3 and np.any(thresholds[3:] >= obs[i,3:]):
                    cloud_mask_MAIA[i] = 0 
                elif (j==1 or j==2) and (DTT_NDxI[i,0] >= 0 or DTT_NDxI[i,1] >= 0):#this is since we used DTT for NDxI
                    pass#cloud_mask_MAIA[i] = 0
                elif np.any(thresholds[0] <= obs[i,0]):#this is for whiteness since 0 is whiter than 1
                    cloud_mask_MAIA[i] = 0
                else:
                    pass

        #compare with CM; cloudy==0, clear==1
        MOD_CM = cloud_mask
        MAIA_CM = cloud_mask_MAIA
        #both return cloudy
        true  = np.where(    (MAIA_CM == 0) & (MOD_CM == 0))[0].sum()
        #both return clear
        false = np.where(    (MAIA_CM == 1) & (MOD_CM != 0))[0].sum()
        #MOD cloudy MAIA clear
        false_neg = np.where((MAIA_CM == 1) & (MOD_CM == 0))[0].sum()
        #MOD clear MAIA cloudy
        false_pos = np.where((MAIA_CM == 0) & (MOD_CM != 0))[0].sum()

        #make result into confusion matrix
        conf_mat = np.array([true, false, false_pos, false_neg])

        #save it back into group/threshold file
        try:
            dataset = hf_thresh.create_dataset('confusion_matrix', data=conf_mat)
            dataset.attrs['label'] = ['both_cloudy, both_clear,\
                                       MOD_cloud_MAIA_clear, MOD_clear_MAIA_cloudy']
        except:
            hf_thresh['confusion_matrix'][:] = conf_mat
        
        #print(threshold_path[-65:-5])
        print((conf_mat[0]+conf_mat[1])/conf_mat.sum())
if __name__ == '__main__':

    import h5py
    import mpi4py.MPI as MPI
    import tables
    import os
    import numpy as np
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            #define paths for the three databases
            home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
            PTA_file_path    = home + 'test_thresholds/'
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
