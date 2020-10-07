import numpy as np
import matplotlib.pyplot as plt
import configparser
import h5py
# 'scene_Accuracy_DOY_bin_{:02d}'.format(DOY_bin)

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA           = config['current PTA']['PTA']
PTA_path      = config['PTAs'][PTA]
Target_Area_X = int(config['Target Area Integer'][PTA])

scene_accur_home = PTA_path + '/' + config['supporting directories']['group_accuracy']
scene_accur_path = scene_accur_home + '/' + 'group_ID_accuracy.h5'
filepath = scene_accur_path


s_list = np.zeros((15,10,12))
num_samples_list=np.zeros((15,10,12))

with h5py.File(filepath, 'r') as hf:
    bins = list(hf.keys())

    for VZA in range(15):
        for SZA in range(10):
            for RAA in range(12):
                #place to store for each bin
                accuracy =[]
                num_samples=[]
                #find group names of bin combo in for loop step
                bin_IDs = [x for x in bins if x[:40]=='confusion_matrix_cosSZA_{:02d}_VZA_{:02d}_RAZ_{:02d}'.format(SZA,VZA,RAA)]
                #cycle through all the unque SVG bins (DOY and SID may change ofcourse)
                for i, bin_ID in enumerate(bin_IDs):
                    print(SZA,VZA,RAA,bin_ID)
                    accuracy.append(hf[bin_ID+'/accuracy'][()])
                    # print(hf[bin_ID+'/accuracy'][()])
                    num_samples.append(hf[bin_ID+'/num_samples'][()])
                s_temp = np.array(accuracy)
                num_samples_temp = np.array(num_samples)
                # s_temp = s_temp[s_temp>=0]
                s_list[VZA,SZA,RAA] = np.nanmean(s_temp)*100
                num_samples_list[VZA,SZA,RAA] = np.nansum(np.array(num_samples))
