import h5py
import numpy
import configparser

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]
thresh_home  = config['supporting directories']['thresh']

thresh_path = '{}/{}/{}'.format(PTA_path, thresh_home, 'thresholds_DOY_041_to_048_bin_05.h5')

with h5py.File(thresh_path, 'r') as hf_thresh:
    DOYs = list(hf_thresh['TA_bin_00'].keys())
    obs  = list(hf_thresh['TA_bin_00/' + DOYs[0]].keys())

    SVI_negative_count = 0
    num_positive_SVI   = 0
    for DOY in DOYs:
        SVI_path = '{}/{}/{}'.format('TA_bin_00', DOY, obs[4])
        SVI = hf_thresh[SVI_path][()]
        num_negative_SVI = len(SVI[SVI<0])
        num_positive_SVI += len(SVI) - num_negative_SVI
        if num_negative_SVI > 0:
            SVI_negative_count += num_negative_SVI

print(num_negative_SVI)
print(num_positive_SVI)
