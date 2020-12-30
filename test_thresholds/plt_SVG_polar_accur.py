import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import configparser
import h5py
# 'scene_Accuracy_DOY_bin_{:02d}'.format(DOY_bin)

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA           = config['current PTA']['PTA']
PTA_path      = config['PTAs'][PTA]
Target_Area_X = int(config['Target Area Integer'][PTA])

group_accur_home = PTA_path + '/' + config['supporting directories']['group_accuracy']
group_accur_path = group_accur_home + '/' + 'group_ID_accuracy.h5'
filepath = group_accur_path


# s_list = np.zeros((10,12,15))
# num_samples_list=np.zeros((10,12,15))
# weighted_accuracy = np.zeros((10,12,15))
# with h5py.File(filepath, 'r') as hf:
#     bins = list(hf.keys())
#
#     for SZA in range(10):
#         for RAA in range(12):
#             for VZA in range(15):
#                 #place to store for each bin
#                 accuracy =[]
#                 num_samples=[]
#                 #find group names of bin combo in for loop step
#                 bin_IDs = [x for x in bins if x[:40]=='confusion_matrix_cosSZA_{:02d}_VZA_{:02d}_RAZ_{:02d}'.format(SZA,VZA,RAA)]
#                 #cycle through all the unque SVG bins (DOY and SID may change ofcourse)
#                 for i, bin_ID in enumerate(bin_IDs):
#                     # print(SZA,VZA,RAA,bin_ID)
#                     accuracy.append(hf[bin_ID+'/accuracy'][()])
#                     # print(hf[bin_ID+'/accuracy'][()])
#                     num_samples.append(hf[bin_ID+'/num_samples'][()])
#                 accuracy, num_samples = np.array(accuracy), np.array(num_samples)
#                 weighted_accuracy[SZA, RAA, VZA] = np.sum(accuracy*num_samples) / np.sum(num_samples)
#         print('SZA bin: ', SZA)
#                 # s_temp = np.array(accuracy)
#                 # num_samples_temp = np.array(num_samples)
#                 # # s_temp = s_temp[s_temp>=0]
#                 # s_list[SZA,RAA,VZA] = np.nanmean(s_temp)*100
#                 # num_samples_list[SZA,RAA,VZA] = np.nansum(np.array(num_samples))
# np.savez('./SVG_accur_data.npz', weighted_accuracy=weighted_accuracy)


#-- Generate Data -----------------------------------------
# Using linspace so that the endpoint of 360 is included...
azimuths = np.radians(np.arange(0,195, 15))
zeniths  = np.arange(0, 75, 5)

r, theta = np.meshgrid(zeniths, azimuths)
# values = np.random.random((azimuths.size, zeniths.size))

data = np.load('./SVG_accur_data.npz')
dataset_names = data.files

weighted_accuracy_SVG = data[dataset_names[0]]*100

# accuracy_SVG = np.moveaxis(accuracy_SVG, 0,1)
# num_smaples_SVG = np.moveaxis(num_smaples_SVG, 0,1)
# print(accuracy_SVG.shape, num_smaples_SVG.shape)

#-- Plot... ------------------------------------------------
fig, ax = plt.subplots(3,2, subplot_kw=dict(projection='polar'))


for i, a in enumerate(ax.flat):
    cmap = cm.get_cmap('plasma', 20)
    a.set_thetagrids(np.arange(0,195,15))

    im = a.pcolormesh(theta, r, weighted_accuracy_SVG[i+4,:,:],\
                      cmap=cmap, vmin=0, vmax=100)
    SZA1 = np.rad2deg(np.arccos((i+3)/10))
    SZA2 = np.rad2deg(np.arccos(((i+3)+1)/10))
    a.set_title('SZA {:2.2f} - {:2.2f} [deg]'.format(SZA1, SZA2))
    a.set_rticks(np.arange(0,75,5))
    a.set_thetamax(180)
    a.grid()

cax = fig.add_axes([0.95, 0.23, 0.02, 0.5])#l,b,w,h
cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0,105,5))

plt.show()
