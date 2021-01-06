import numpy as np
import h5py

home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/results/conf_matx_group/numKmeansSID_16/'
master_conf_matx = np.zeros((4))
for DOY_bin in range(46):
    grouped_path = home + 'conf_matx_group_DOY_bin_{:02d}.h5'.format(DOY_bin)
    with h5py.File(grouped_path, 'r') as hf_group:
        group_keys = list(hf_group.keys())
        for group in group_keys:
            conf_mat_group_temp = hf_group[group][()]
            master_conf_matx += conf_mat_group_temp
    print(DOY_bin)
print(master_conf_matx)

    # print(group_keys[-1])
    # print(len(group_keys))
    # print(data.shape)
