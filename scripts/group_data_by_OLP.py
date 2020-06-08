def group_data(OLP, obs, CM, hf_group):
    """
    Objective:
        Group data by observable_level_paramter (OLP), such that all data in same
        group has a matching OLP. The data point is stored in the group with its
        observables, cloud mask, time stamp, and lat/lon. Will process one MAIA
        grid at a time. No return, will just be written to file in this function.
    Return:
        void
    """
    # observable_level_parameter = np.dstack((binned_cos_SZA ,\
    #                                         binned_VZA     ,\
    #                                         binned_RAZ     ,\
    #                                         Target_Area    ,\
    #                                         land_water_mask,\
    #                                         snow_ice_mask  ,\
    #                                         sfc_ID         ,\
    #                                         binned_DOY     ,\
    #                                         sun_glint_mask))

    OLP = OLP.reshape(1000**2, 6)
    obs = obs.reshape(1000**2, 7)
    CM  = CM.reshape(1000**2)

    #remove empty data points
    #where cos(SZA) is negative (which is not possible)
    full_idx = np.where(OLP[:,0] !=-999) # obs -> (1, 1e6-x)

    OLP = OLP[full_idx[0], :]
    obs = obs[full_idx[0], :]
    CM  = CM[full_idx[0]]

    thresh_dict = {}

    #track number of groups seen to compare to groups actually processed
    # num_groups = 0

    #now for any OLP combo, make a group and save the data points into it
    for i in range(CM.shape[0]):
        #0 cosSZA
        #1 VZA
        #2 RAZ
        #3 TA
        #4 Scene_ID
        #5 DOY
        temp_OLP = OLP[i,:]
        group = 'cosSZA_{:02d}_VZA_{:02d}_RAZ_{:02d}_TA_{:02d}_sceneID_{:02d}_DOY_{:02d}'\
                .format(temp_OLP[0], temp_OLP[1], temp_OLP[2],\
                        1, temp_OLP[4], temp_OLP[5])

        data = np.array([CM[i]   ,\
                         obs[i,0],\
                         obs[i,1],\
                         obs[i,2],\
                         obs[i,3],\
                         obs[i,4],\
                         obs[i,5],\
                         obs[i,6] ])

        #add key with empty list, then populate it.
        thresh_dict.setdefault(group, [])
        thresh_dict[group].append(data)

    for key, val in thresh_dict.items():
        try:
            hf_group.create_dataset(key, data=np.array(val), maxshape=(None,8))
            # num_groups += 1
        except:

            group_shape = hf_group[key].shape[0]
            hf_group[key].resize(group_shape + np.array(val).shape[0], axis=0)
            hf_group[key][group_shape:, :] = np.array(val)

        # print(key[10:16])
if __name__ == '__main__':

    import numpy as np
    import h5py
    import mpi4py.MPI as MPI
    import tables
    import os
    import sys
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:
            file_select = r

            home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'

            DOY_bin = int(sys.argv[1])
            DOY_end = (DOY_bin+1)*8
            DOY_start = DOY_end - 7

            #define paths for the three databases
            PTA_file_path = home + 'LA_database_60_cores/'
            database_files = os.listdir(PTA_file_path)
            database_files = [PTA_file_path + filename for filename in database_files]
            database_files = np.sort(database_files)
            hf_database_path = database_files[file_select]

            PTA_file_path = home + 'observables_database_60_cores/'
            database_files = os.listdir(PTA_file_path)
            database_files = [PTA_file_path + filename for filename in database_files]
            database_files = np.sort(database_files)
            hf_observables_path = database_files[file_select]

            PTA_file_path = home + 'OLP_database_60_cores/'
            database_files = os.listdir(PTA_file_path)
            database_files = [PTA_file_path + filename for filename in database_files]
            database_files = np.sort(database_files)
            hf_OLP_path    = database_files[file_select]

            observables = ['WI', 'NDVI', 'NDSI', 'visRef', 'nirRef', 'SVI', 'cirrus']

            #get data for input into grouping function
            with h5py.File(hf_observables_path       , 'r') as hf_observables,\
                 h5py.File(hf_database_path          , 'r') as hf_database   ,\
                 h5py.File(hf_OLP_path               , 'r') as hf_OLP        :
                hf_database_keys = list(hf_database.keys())
                #grab only DOY bin 6 since I dont have sfc ID yet for other days
                #hf_database_keys = [x for x in hf_database_keys if int(x[4:7])>=48 and int(x[4:7])<=55]
                hf_database_keys = [x for x in hf_database_keys if int(x[4:7])>=DOY_start and int(x[4:7])<=DOY_end]
                #open file to write groups to
                hf_group_path = home + 'group_60_cores/grouped_data_DOY_{:03d}_to_{:03d}_bin_{:02d}_rank_{:02d}.hdf5'.format(DOY_start, DOY_end, DOY_bin, rank)

                # try:
                with h5py.File(hf_group_path, 'w') as hf_group:
                    for time_stamp in hf_database_keys:

                        CM  = hf_database[time_stamp + '/cloud_mask/Unobstructed_FOV_Quality_Flag'][()]
                        OLP = hf_OLP[time_stamp + '/observable_level_paramter'][()]

                        obs_data = np.empty((1000,1000,7), dtype=np.float)
                        for i, obs in enumerate(observables):
                            data_path = '{}/{}'.format(time_stamp, obs)
                            obs_data[:,:,i] = hf_observables[data_path][()]
                        group_data(OLP, obs_data, CM, hf_group)


                # except:
                #     with h5py.File(hf_group_path, 'r+')  as hf_group:
                #         for time_stamp in hf_database_keys:
                #
                #             CM  = hf_database[time_stamp + '/cloud_mask/Unobstructed_FOV_Quality_Flag'][()]
                #             OLP = hf_OLP[time_stamp + '/observable_level_paramter'][()]
                #
                #             obs_data = np.empty((1000,1000,7), dtype=np.float)
                #             for i, obs in enumerate(observables):
                #                 data_path = '{}/{}'.format(time_stamp, obs)
                #                 obs_data[:,:,i] = hf_observables[data_path][()]
                #             group_data(OLP, obs_data, CM, hf_group)
