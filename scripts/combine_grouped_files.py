import os
import h5py

def group_bins(home, group_dir, common_file):
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
    database_files = [path filename for filename in grouped_files]

    #look through each file and find unique bin IDs. Add data to dictiionary
    #After looking through the entire data set, write to one common file

    #add key with empty list, then populate it.
    group_dict = {}

    for group in database_files:
        group_keys = list(group.keys())

        for key in group_keys():
            group_dict.setdefault(key, [])
            group_dict[key].append(data)

    with h5py.File(home + common_file, 'w') as hf_group:
        for key, val in group_dict.items():
            try:
                hf_group.create_dataset(key, data=np.array(val), maxshape=(None,8))

            except:
                group_shape = hf_group[key].shape[0]
                hf_group[key].resize(group_shape + np.array(val).shape[0], axis=0)
                hf_group[key][group_shape:, :] = np.array(val)

if __name__ == '__main__':

    home        = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
    group_dir   = 'group_DOY_05_60_cores'
    common_file = 'grouped_obs_and_CM.hdf5'
    group_bins(home, group_dir, common_file)
