import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import configparser
import h5py

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA           = config['current PTA']['PTA']
PTA_path      = config['PTAs'][PTA]
Target_Area_X = int(config['Target Area Integer'][PTA])

group_accur_home = PTA_path + '/' + config['supporting directories']['group_accuracy']
group_accur_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/results/group_accuracy'
group_accur_path = group_accur_home + '/' + 'group_ID_accuracy.h5'
filepath = group_accur_path

def plot_accur_by_SID():
    plt.rcParams['font.size'] = 16

    s_list = []
    num_samples_list=[]

    with h5py.File(filepath, 'r') as hf:
        bins = list(hf.keys())

        for j in range(20):
            accuracy =[]
            num_samples=[]
            for i, bin_ID in enumerate(bins):
                # print(bin_ID)
                if int(bin_ID[-9:-7]) == j :
                    accuracy.append(hf[bin_ID+'/accuracy'][()])
                    # print(hf[bin_ID+'/accuracy'][()])
                    num_samples.append(hf[bin_ID+'/num_samples'][()])
            s_temp = np.array(accuracy)
            num_samples_temp = np.array(num_samples)
            # s_temp = s_temp[s_temp>=0]
            avg_weighted_accur = np.nansum(s_temp*num_samples_temp)/np.nansum(num_samples_temp)
            s_list.append(avg_weighted_accur*100)
            num_samples_list.append(np.nansum(num_samples_temp))
    print(s_list)
    print(num_samples_list)
    print(np.array(s_list).shape)
    x = np.arange(20)

    #now plot the histogram
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }
    plt.rcParams['font.size'] = 16

    fig, ax = plt.subplots()

    fig.suptitle('Accuracy by Surface Type LA PTA')

    x_ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'coast','water', 'glint','snow']

    color = 'tab:pink'
    ax.set_xlabel('Surface Type')
    ax.set_ylabel('% Accuracy', color=color)
    # ax.set_ylim(75, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.scatter(x, s_list, color=color)
    ax.plot(x, s_list, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_yticks(np.arange(40,105,5))
    ax.grid()

    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:cyan'
    ax1.set_ylabel('number of samples', color=color)  # we already handled the x-label with ax1
    ax1.semilogy(x, num_samples_list, color=color)
    ax1.set_yticks([10**9, 10**10, 10**11, 10**12, 10**13])

    # ax1.set_ylim(0, 10**12)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.show()

def plot_accur_by_DOY():
    plt.rcParams['font.size'] = 16

    s_list = []
    num_samples_list=[]

    with h5py.File(filepath, 'r') as hf:
        bins = list(hf.keys())

        for j in range(46):
            accuracy =[]
            num_samples=[]
            for i, bin_ID in enumerate(bins):
                # print(bin_ID)
                if int(bin_ID[-2:]) == j :
                    accuracy.append(hf[bin_ID+'/accuracy'][()])
                    # print(hf[bin_ID+'/accuracy'][()])
                    num_samples.append(hf[bin_ID+'/num_samples'][()])
            s_temp = np.array(accuracy)
            num_samples_temp = np.array(num_samples)
            # s_temp = s_temp[s_temp>=0]
            avg_weighted_accur = np.nansum(s_temp*num_samples_temp)/np.nansum(num_samples_temp)
            s_list.append(avg_weighted_accur*100)
            num_samples_list.append(np.nansum(num_samples_temp))
    print(s_list)
    print(num_samples_list)
    print(np.array(s_list).shape)
    x = np.arange(46)

    #now plot the histogram
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots()

    fig.suptitle('Accuracy by DOY LA PTA 2002-2019')

    x_ticks = np.arange(1,47)*8
    x_ticks[-1] = 365
    color = 'tab:pink'
    ax.set_xlabel('Julian DOY (valid previous 8 days)')
    ax.set_ylabel('% Accuracy', color=color)
    # ax.set_ylim(75, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.scatter(x, s_list, color=color)
    ax.plot(x, s_list, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_yticks(np.arange(87,101,1))
    ax.grid()

    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:cyan'
    ax1.set_ylabel('number of samples', color=color)  # we already handled the x-label with ax1
    ax1.semilogy(x, num_samples_list, color=color)
    # yticks = np.arange(1,11)*10**11
    # ax1.set_yticks(yticks)
    # ax1.set_yticklabels(yticks, style='sci')

    ax1.set_ylim(10**11, 10**12)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.show()

plot_accur_by_DOY()
