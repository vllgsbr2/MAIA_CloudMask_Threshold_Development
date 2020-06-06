import os
import h5py
import numpy as np
import sys

def group_bins(home, group_dir, common_file, DOY_bin, rank):
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
    path = home + group_dir
    grouped_files  = np.sort(os.listdir(path))
    database_files = [path +'/'+ filename for filename in grouped_files]

    if len(database_files) != 46:
        print('group by OLP failed to produce all DOY bins')
        sys.exit()

    #look through each file and find unique bin IDs. Add data to dictiionary
    #After looking through the entire data set, write to one common file

    #add key with empty list, then populate it.
    group_dict = {}

    for i in range(46):
        group = grouped_files = home + 'grouped_data_DOY_{:03d}_to_{:03d}_bin_{:02d}_rank_{:02d}.hdf5'.format(DOY_start, DOY_end, DOY_bin, i)
        with h5py.File(group, 'r') as hf_group_:
            group_keys = list(hf_group_.keys())

            if len(group_keys) > 0:
                for key in group_keys:
                    data = hf_group_[key][()]
                    group_dict.setdefault(key, [])
                    group_dict[key].append(data)

    if not os.path.exists(home + 'grouped_obs_and_CMs/'  + common_file):
        with h5py.File(home + 'grouped_obs_and_CMs/'  + common_file, 'w') as hf_group:
            for key, val in group_dict.items():
                for arr in val:
                    try:
                        hf_group.create_dataset(key, data=np.array(arr), maxshape=(None,8))

                    except:
                        group_shape = hf_group[key].shape[0]
                        hf_group[key].resize(group_shape + np.array(arr).shape[0], axis=0)
                        hf_group[key][group_shape:, :] = np.array(arr)
    #print('done')
if __name__ == '__main__':

    import mpi4py.MPI as MPI
    import tables
    import sys
    import os
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            home        = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
            group_dir   = 'group_60_cores'
            DOY_bin     = int(sys.argv[1])
            DOY_end     = (DOY_bin+1)*8
            DOY_start   =  DOY_end - 7
            common_file = 'grouped_obs_and_CM_{:03d}_to_{:03d}_bin_{:02d}.hdf5'.format(DOY_start, DOY_end, DOY_bin)
            group_bins(home, group_dir, common_file, DOY_bin, rank)
