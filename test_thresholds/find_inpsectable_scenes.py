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

    with open('./scenes_worth_inspecting.txt', 'w') as txt_good_scenes:
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

                    num_fill_vals    = np.where(data != -999)[0].shape[0]
                    image_shape      = data.shape
                    num_pix_total    = image_shape[0]*image_shape[1]
                    percent_bad_pix  = num_fill_vals/num_pix_total
                    percent_good_pix = 1 - percent_bad_pix

                    if percent_good_pix > 0.1:
                        metadata += '{} , {:1.2f}\n'.format(scene, percent_good_pix)
                        count_good_scenes += 1

                txt_good_scenes.write(metadata)
            print(count_good_scenes)

if __name__ == '__main__':
    find_good_scenes()
