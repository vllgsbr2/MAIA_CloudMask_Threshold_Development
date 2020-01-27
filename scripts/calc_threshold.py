def calc_thresh(group_file):
    '''
    Objective:
        Takes in grouped_data_cosSZA_08_VZA_09_RAZ_08_TA_01_DOY_05_sceneID_09.hdf5
        files. Inside are a data points containing the cloud mask and observables
        for that pixel. The OLP is in the file name. The threshold is then
        calculated for that group and saved into a threshold file
    Arguments:
        group_file {str} -- contains data points to calc threshold for that OLP
    Return:
        void
    '''
    #remember to change Ref to BRF
    #^ unfortunately have to redo observable files and then the grouped files

    with h5py.File(group_file, 'r') as hf_group:
        hf_keys    = list(hf_group.keys())
        num_points = len(hf_keys)
        cloud_mask = np.zeros((num_points))

        observables = ['WI', 'NDVI', 'NDSI', 'visRef', 'nirRef', 'SVI', 'cirrus']
        obs = np.empty((num_points, 7))
        Thresholds = np.zeros((num_points, 7))

        for i, data_point in enumerate(hf_keys):
            data = hf_group[data_point + 'label_and_obs'][()]

            cloud_mask[i] = int(data[0])
            obs[i]        = data[1:]

        for i in range(7):
            clear_idx = np.where(cloud_mask != 0)
            clear_obs = obs[clear_idx,:]

            thresholds[:,i] = np.percentile(clear_obs[:,i], 1)

        thresh_name = 'threshold_{}'.format(group_file[13:])

        try:
            dataset = hf_group.create_dataset('thresholds', data=thresholds)

            #label the data
            dim_names = ['WI', 'NDVI', 'NDSI', 'visRef',\
                         'nir_Ref', 'SVI', 'cirrus']
            for i, name in enumerate(dim_names):
                dataset.dims[i].label = name
        except:
            hf_group['thresholds'][:] = thresholds


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
            PTA_file_path    = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/grouped_data'
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
                calc_thresh(path)
