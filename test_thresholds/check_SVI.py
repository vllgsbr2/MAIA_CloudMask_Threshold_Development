import h5py
import numpy as np
import configparser
import os
import sys

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]
thresh_home  = config['supporting directories']['thresh']

# thresh_path = '{}/{}/{}'.format(PTA_path, thresh_home, 'thresholds_DOY_041_to_048_bin_05.h5')

thresh_path = '{}/{}/'.format(PTA_path, thresh_home)
thresh_files = [thresh_path + x for x in os.listdir(thresh_path)]

fill_val = -999

for thresh_file in thresh_files:
    with h5py.File(thresh_file, 'r') as hf_thresh:
        DOYs = list(hf_thresh['TA_bin_00'].keys())
        obs  = list(hf_thresh['TA_bin_00/' + DOYs[0]].keys())

        num_negative_SVI = 0
        num_positive_SVI = 0
        for DOY in DOYs:
            SVI_path = '{}/{}/{}'.format('TA_bin_00', DOY, obs[4])
            SVI = hf_thresh[SVI_path][()].flatten()

            # SVI = hf_thresh[SVI_path][()]
            # print(np.where((SVI<0) & (SVI != fill_val)))
            # SVI = SVI.flatten()

            neg_SVI = SVI[(SVI<0) & (SVI != fill_val)]
            num_negative_SVI += len(neg_SVI)
            num_positive_SVI += len(SVI[SVI>=0])



    print('%neg {:1.6f}, num neg {:05d}, num pos {:05d}, neg SVIs {}'.format(num_negative_SVI/num_positive_SVI, num_negative_SVI, num_positive_SVI, neg_SVI))
