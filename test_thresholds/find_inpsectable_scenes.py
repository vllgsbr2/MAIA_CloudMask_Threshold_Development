def find_good_scenes():
    import os
    import numpy as np
    import h5py
    import configparser

    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config           = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA          = config['current PTA']['PTA']
    PTA_path     = config['PTAs'][PTA]

    data_home          = '{}/{}/'.format(PTA_path, config['supporting directories']['Database'])
    database_files     = os.listdir(data_home)
    database_filepaths = [data_home + x for x in database_files]

    with open('./scenes_2_inspecting.txt', 'w') as txt_good_scenes:
        txt_good_scenes.write('time_stamp YYYYDDD.HHMM, % valid pixels\n')
        for db_file in database_filepaths:
            print(db_file[-30:])
            metadata = ''
            with h5py.File(db_file, 'r') as hf_file:
                time_stamps = list(hf_file.keys())
                count_good_scenes = 0
                for scene in time_stamps:
                    rad_path = '{}/{}/{}'.format(scene, 'radiance', 'band_1')
                    data     = hf_file[rad_path][()]

                    num_fill_vals    = np.where(data == -999)[0].shape[0]
                    image_shape      = data.shape
                    num_pix_total    = image_shape[0]*image_shape[1]
                    percent_bad_pix  = num_fill_vals/num_pix_total
                    percent_good_pix = 1 - percent_bad_pix

                    if percent_good_pix > 0.1:
                        metadata += '{} , {:1.2f}\n'.format(scene, percent_good_pix)
                        count_good_scenes += 1

                txt_good_scenes.write(metadata)
            print(count_good_scenes)

def choose_random_scenes():
    import pandas as pd
    import numpy as np

    scenes_2_inspect = []
    df_scenes   = pd.read_csv('./scenes_2_inspecting.txt', header=0, delimiter=',', dtype=str)
    time_stamp = list(df_scenes.columns)[0]
    time_stamps = df_scenes[time_stamp].values

    low, high = 0, time_stamps.shape[0]-1
    size = 100
    random_scene_idx = np.random.randint(low=low, high=high, size=size)
    rand_scenes = time_stamps[random_scene_idx]
    rand_scenes_str = ''

    with open('./scenes_worth_inspecting.txt', 'w') as txt_good_scenes:
        for scene in rand_scenes:
            rand_scenes_str += '{}\n'.format(scene)

        txt_good_scenes.write(rand_scenes_str)

def graph_scenes(scenes_file):
    '''
    scenes_file {str} -- txt file where each row is time stamp YYYYDDD.HHMM
                         i.e. 2020.048.1230 -> DOY 48, year 2020, 12:30 UTC
    '''

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from rgb_enhancement import get_enhanced_RGB
    import configparser
    import os
    import pandas as pd

    #grab relevant  scenes
    df_scenes   = pd.read_csv('./scenes_worth_inspecting.txt', header=None, dtype=str, delimiter='\n')
    scenes = [x[:-1] for x in df_scenes.values[:,0]]

    # with open(scenes_file, 'r') as txt_scenes:
    #     scenes = []
    #     for time_stamp in txt_scenes:
    #         scenes.append(time_stamp)

    #find output files
    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config           = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA          = config['current PTA']['PTA']
    PTA_path     = config['PTAs'][PTA]

    #grab MCM and RGB from MCM Output files
    output_home      = '{}/{}/'.format(PTA_path, config['supporting directories']['MCM_Output'])
    output_dir       = os.listdir(output_home)
    #use only scenes from scene file
    output_dir       = [x for x in output_dir if x in scenes]
    output_filepaths = [output_home + x + '/MCM_Output.h5' for x in output_dir]
    time_stamps      = output_dir

    # #grab MOD35 from here
    # data_home          = '{}/{}/'.format(PTA_path, config['supporting directories']['Database'])
    # database_files     = os.listdir(data_home)
    # database_filepaths = [data_home + x for x in database_files]

    for MCM_Output, time_stamp in zip(output_filepaths, time_stamps):
        print(time_stamp)
        #get RGB and MCM
        with h5py.File(MCM_Output, 'r') as hf_MCM_Output:
            # R_band_4  = hf_MCM_Output['Reflectance/band_04'][()]
            # R_band_5  = hf_MCM_Output['Reflectance/band_05'][()]
            R_band_6  = hf_MCM_Output['Reflectance/band_06'][()]
            R_band_13 = hf_MCM_Output['Reflectance/band_13'][()]

            MCM = hf_MCM_Output['cloud_mask_output/final_cloud_mask'][()]

        #construct and enhance RGB
        print(np.where(R_band_6 == -999)[0].shape[0])
        if np.where(R_band_6 == -999)[0].shape[0] <= 12000:
            # RGB            = np.dstack((R_band_6, R_band_5, R_band_4))
            # RGB[RGB==-999] = 0
            # RGB            = get_enhanced_RGB(RGB)

            #normalize band 6 and band 13
            # R_band_6_norm = (R_band_6 - R_band_6.mean()) / R_band_6.std()
            # R_band_13_norm = (R_band_13 - R_band_13.mean()) / R_band_13.std()
            # R_band_13_6_norm = R_band_13_norm + R_band_6_norm
            #then add them together

            #plot RGB enhanced against MCM binary
            f, ax = plt.subplots(ncols=2)

            image_MCM = ax[0].imshow(MCM, cmap='binary', vmin=0, vmax=1.1)
            ax[1].imshow(R_band_6 + 3*R_band_13, cmap = 'bone')

            ax[0].set_title('MCM ' + time_stamp)
            ax[1].set_title('RGB ' + time_stamp)

            image_MCM.cmap.set_over('red')

            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

            plt.show()
        else:
            print('no good pixels what the heck')


if __name__ == '__main__':
    # find_good_scenes()
    # choose_random_scenes()
    scenes_file = './scenes_worth_inspecting.txt'
    graph_scenes(scenes_file)
