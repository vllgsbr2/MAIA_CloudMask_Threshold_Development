import h5py
import numpy as np
from rgb_enhancement import get_enhanced_RGB
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as matCol
from matplotlib.colors import ListedColormap
import os
import configparser

#read in config file
config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

#grab output files
MCM_output_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/results/MCM_Output/Guangyu_output_dec_1_2020/'
time_stamps     = np.sort(os.listdir(MCM_output_home))

#grab input files
MCM_input_home = '/data/gdi/c/gzhao1/MCM-thresholds/PTAs/LosAngeles/MCM_Input/'
test_data_JPL_paths = os.listdir(MCM_input_home)
time_stamps         = [x[14:26] for x in test_data_JPL_paths]
test_data_JPL_paths = [MCM_input_home + x for x in test_data_JPL_paths]

for time_stamp, test_data_JPL_path in zip(time_stamps, test_data_JPL_paths):
    output_file_path = MCM_output_home + time_stamp + '/'
    with h5py.File(output_file_path, 'r') as hf_MCM_output:
        DTT = hf_MCM_output['cloud_mask_output/DTT'][()]
        MCM = hf_MCM_output['cloud_mask_output/DTT'][()]
        SID = hf_MCM_output['cloud_mask_output/DTT'][()]

        #get RGB
        R_red = hf_MCM_output['Reflectance/band_06'][()]
        R_grn = hf_MCM_output['Reflectance/band_05'][()]
        R_blu = hf_MCM_output['Reflectance/band_04'][()]

        RGB = np.dstack((R_red, R_grn, R_blu))
        RGB[RGB==-999] = 0
        RGB = get_enhanced_RGB(RGB)


    #grab mod35 cm from input file
    with h5py.File(test_data_JPL_path, 'r') as hf_output:
        mod35cm = hf_output['MOD35_cloud_mask'][()]

    #plot
    #DTT_WI, DTT_NDVI, DTT_NDSI, DTT_VIS_Ref, DTT_NIR_Ref, DTT_SVI, DTT_Cirrus
    obs_namelist = ['WI', 'NDVI', 'NDSI', '0.65µm BRF', '0.86µm', 'SVI', 'Cirrus']
    f, ax = plt.subplots(nrows=2, ncols=7)

    for i, a in enumerate(ax.flat):

        #plot DTT first
        if i < 7:
            a.imshow(DTT[:,:,i])
            a.set_title(obs_namelist[i])

        #plot BRF/MOD35/MCM/SID
        if i==7:
            a.imshow(RGB)
            a.set_title('RGB')
        if i==8:
            a.imshow(mod35cm)
            a.set_title('MOD35')
        if i==9:
            a.imshow(MCM)
            a.set_title('RGB')
        if i==10:
            a.imshow(SID)
            a.set_title('RGB')

        #turn off unused axes
        if i >= 11:
            a.axis('off')
