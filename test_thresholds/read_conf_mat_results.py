import pandas as pd
import h5py

data = pd.read_csv('./conf_mat_results.csv')

#now get group name
#define paths for the three databases
home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/results/conf_matx_group/numKmeansSID_16/'

for DOY_bin in range(46):
    grouped_path = home + 'conf_matx_group_DOY_bin_{:02d}.h5'.format(DOY_bin)
    with h5py.File(grouped_path, 'r') as hf_group:
        group_keys = list(hf_group.keys())
    print(group_keys[0])
    print(len(group_keys))
    print(data.shape)
