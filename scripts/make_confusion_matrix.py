def confusion_matrix(threshold_path):

    with h5py.File(threshold_path, 'r+') as hf_thresh:

        hf_keys    = list(hf_thresh.keys())

        n=111 # sza, vza, raz, scene_id
        #print(threshold_path[n+7:n+9])
        OLP =[int(threshold_path[n+7:n+9])  ,\
              int(threshold_path[n+14:n+16]),\
              int(threshold_path[n+21:n+23]),\
              int(threshold_path[n+45:n+47]) ]

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
            #put 0.001 instead of zero to avoid errors
            if T==0:
                T = 1e-3
            DTT_NDxI[:,i-1] = (T - np.abs(NDxI)) / T

        #see if any obs trigger cloudy
        #assume clear (i.e. 1), cloudy is 0
        cloud_mask_MAIA = np.ones((num_points))

        for i in range(num_points):
            #[WI_0, NDVI_1, NDSI_2, VIS_3, NIR_4, SVI_5, Cirrus_6]
            #water = 30 / sunglint over water = 31/ snow =32 / land = 0-11
            for j in range(7):
                #this is for whiteness since 0 is whiter than 1 and not applied over snow/ice or sunglint
                if j==0 and thresholds[0] <= obs[i,0] and (OLP[-1]!=31 and OLP[-1]!=32):
                    cloud_mask_MAIA[i] = 0
                #DTT for NDxI. Must exceed 0
                #NDVI everything but snow
                elif j==1 and OLP[-1]!=32 and DTT_NDxI[i,0] >= 0:
                    cloud_mask_MAIA[i] = 0
                #NDSI only over snow
                elif j==2 and OLP[-1]==32 and DTT_NDxI[i,1] >= 0:
                    cloud_mask_MAIA[i] = 0
                #VIS, NIR, Cirrus. Must exceed thresh
                #VIS applied only over land
                elif j==3 and OLP[-1]<=11 and thresholds[3] >= obs[i,3]:
                    cloud_mask_MAIA[i] = 0
                #NIR only applied over water (no sunglint)
                elif j==4 and OLP[-1]==30 and thresholds[4] >= obs[i,4]:
                    cloud_mask_MAIA[i] = 0
                #SVI applied over all surfaces when over 0. Must exceed thresh
                elif j==5 and obs[i,5] >= 0.0  and thresholds[5] >= obs[i,5]:
                    cloud_mask_MAIA[i] = 0
                #j==5 is SVI. So this is j==6 for cirrus applied everywhere
                elif j==6 and thresholds[6] >= obs[i,6]:
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
        print('{},{},{},{}'.format((conf_mat[0]+conf_mat[1])/conf_mat.sum(), conf_mat[0], conf_mat[1], conf_mat.sum()))
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
