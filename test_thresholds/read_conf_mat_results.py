import pandas as pd
import h5py 

data = pd.read_csv('./conf_mat_results.csv')

#now get group name
#define paths for the three databases
home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
grouped_path = home + 'grouped_obs_and_CM.hdf5'

with h5py.File(grouped_path, 'r') as hf_group:

    group_keys = list(hf_group.keys())

print(len(group_keys))
print(data.shape)
