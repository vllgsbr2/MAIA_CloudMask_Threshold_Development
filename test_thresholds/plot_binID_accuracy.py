import h5py
import numpy as np
import os
import tables
from sklearn.metrics import f1_score
#from sklearn.metrics import jaccard_score as IOU <- for scene level not bin level
import matplotlib.pyplot as plt

tables.file._open_files.close_all()

#grab confusion matrix from threshold file
#calculate the accuracy
#make ordered histogram with line showing number of samples over it
home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
thresh_path  = home + 'thresholds_MCM_efficient.hdf5'

with h5py.File(thresh_path, 'r') as hf_thresh:
    bins = list(hf_thresh['TA_bin_01/DOY_bin_06'].keys())

    #2 accuracies -> [combined, cloud, clear, f1 score]
    accuracy = np.zeros((len(bins), 4))
    num_samples = np.zeros((len(bins)))

    for i, bin_ID in enumerate(bins):
        path = 'TA_bin_01/DOY_bin_06/confusion_matrix_{}'.format(bin_ID)
        conf_matx = hf_thresh[path][()]

        num_samples[i] = conf_matx.sum()

        #both_cloudy, both_clear, MOD_cloud_MAIA_clear, MOD_clear_MAIA_cloudy

        #pred/truth
        #cloudy cloudy
        a_pred = np.zeros((conf_matx[0]))
        a_true = np.zeros((conf_matx[0]))
        #clear clear
        d_pred = np.ones((conf_matx[1]))
        d_true = np.ones((conf_matx[1]))
        #cloudy clear
        b_pred = np.zeros((conf_matx[3]))
        b_true = np.ones((conf_matx[3]))
        #clear cloudy
        c_pred = np.ones((conf_matx[2]))
        c_true = np.zeros((conf_matx[2]))

        y_true = np.concatenate((a_true, b_true, c_true, d_true), axis=0)
        y_pred = np.concatenate((a_pred, b_pred, c_pred, d_pred), axis=0)

        #4 accuracies -> [combined, cloud, clear, f1 score]
        accuracy[i, 0] = conf_matx[0] + conf_matx[1] / conf_matx.sum()
        accuracy[i, 1] = conf_matx[0] / (conf_matx[3] + conf_matx[0])
        accuracy[i, 2] = conf_matx[1] / (conf_matx[2] + conf_matx[1])
        accuracy[i, 3] = f1_score(y_true, y_pred)
        print('{:2.2f}  |  '.format(accuracy[i,0]*100, bin_ID))
    #now plot the histogram
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 18,
            }
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots()

    # fig.suptitle('Zonal Mean of Mean Meridional Specific Humidity Flux')

    color = 'tab:pink'
    ax.set_xlabel('bin ID')
    ax.set_ylabel('% Accuracy', color=color)
    ax.plot(bins, accuracy[:, 0], color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:cyan'
    ax1.set_ylabel('number of samples', color=color)  # we already handled the x-label with ax1
    ax1.plot(bins, num_samples, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.show()
