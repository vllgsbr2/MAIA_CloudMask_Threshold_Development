import os
import h5py
import numpy as np
import sys

def group_bins(home, group_dir, common_file, DOY_bin):
    '''
    for easy parallelization and file writting over head, the bins were
    written into a file according to the processor that analyzed it. So
    we must look through all grouped files and combine like bins. This
    way a threshold can be calculated easily and in a highly
    parallelized manner.
    '''

    #first list all files in directory
    DOY_end = (DOY_bin+1)*8
    DOY_start = DOY_end - 7
    grouped_files  = np.sort(os.listdir(group_dir))
    database_files = ['{}/{}'.format(group_dir, filename) for filename in grouped_files]

    # if len(database_files) != 46:
    #     print('group by OLP failed to produce all DOY bins')
    #     sys.exit()

    #look through each file and find unique bin IDs. Add data to dictiionary
    #After looking through the entire data set, write to one common file

    #add key with empty list, then populate it.
    group_dict = {}

    for i in range(60):
        group =  '{}/grouped_data_DOY_{:03d}_to_{:03d}_bin_{:02d}_rank_{:02d}.h5'\
                        .format(group_dir, DOY_start, DOY_end, DOY_bin, i)
        print(group)
        with h5py.File(group, 'r') as hf_group_:
            group_keys = list(hf_group_.keys())

            if len(group_keys) > 0:
                for key in group_keys:
                    data = hf_group_[key][()]
                    group_dict.setdefault(key, [])
                    group_dict[key].append(data)

    # #keep trying to gain access until it is available
    # #once access is gained, it should write to file, and then exit while loop
    # being_accessed = True
    # while being_accessed:
    #     try:
    with h5py.File(common_file, 'w') as hf_group:
        for key, val in group_dict.items():
            for arr in val:
                try:
                    hf_group.create_dataset(key, data=np.array(arr), maxshape=(None,8))

                except:
                    group_shape = hf_group[key].shape[0]
                    hf_group[key].resize(group_shape + np.array(arr).shape[0], axis=0)
                    hf_group[key][group_shape:, :] = np.array(arr)
        #     being_accessed = False
        # except Exception as e:
        #     print(e)
        #     being_accessed = True
    #print('done')
if __name__ == '__main__':

    import mpi4py.MPI as MPI
    import configparser
    # import tables
    import sys
    import os
    #tables.file._open_files.close_all()

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

            group_dir    = '{}/{}'.format(PTA_path, config['supporting directories']['group_intermediate'])
            combined_dir = '{}/{}'.format(PTA_path, config['supporting directories']['combined_group'])
            DOY_bin      = r#int(sys.argv[1])
            DOY_end      = (DOY_bin+1)*8
            DOY_start    =  DOY_end - 7
            common_file  = '{}/grouped_obs_and_CM_{:03d}_to_{:03d}_bin_{:02d}.h5'.format(combined_dir, DOY_start, DOY_end, DOY_bin)
            group_bins(home, group_dir, common_file, DOY_bin)
