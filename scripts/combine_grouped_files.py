import os
import h5py
import numpy as np
import time
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
    path = home + group_dir
    grouped_files  = os.listdir(path)
    database_files = [path +'/'+ filename for filename in grouped_files]
    database_file = database_files[DOY_bin]

    #look through each file and find unique bin IDs. Add data to dictiionary
    #After looking through the entire data set, write to one common file

    #add key with empty list, then populate it.
    group_dict = {}

    #for group in database_files:
    with h5py.File(database_file, 'r') as hf_group_:
        group_keys = list(hf_group_.keys())

        for key in group_keys:
            data = hf_group_[key][()]
            group_dict.setdefault(key, [])
            group_dict[key].append(data)

    with h5py.File(home + 'grouped_obs_and_CMs/'  + common_file, 'w') as hf_group:
        for key, val in group_dict.items():
            for arr in val:
                try:
                    hf_group.create_dataset(key, data=np.array(arr), maxshape=(None,8))

                except:
                    group_shape = hf_group[key].shape[0]
                    hf_group[key].resize(group_shape + np.array(arr).shape[0], axis=0)
                    hf_group[key][group_shape:, :] = np.array(arr)
    print('done')
if __name__ == '__main__':

    import mpi4py.MPI as MPI
    import tables
    import sys
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:
 
            home        = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
            group_dir   = 'group_60_cores'
            DOY_bin = r
            DOY_end = (DOY_bin+1)*8
            DOY_start = DOY_end - 7
            common_file = 'grouped_obs_and_CM_{:03d}_to_{:03d}.hdf5'.format(DOY_start, DOY_end, DOY_bin) 
            group_bins(home, group_dir, common_file, DOY_bin)
