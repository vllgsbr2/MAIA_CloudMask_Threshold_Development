def get_MODIS_file_paths(MOD02_txt, MOD03_txt, MOD35_txt):
    import numpy as np

    with open(MOD02_txt, 'r') as txt_MOD02_files,\
         open(MOD03_txt, 'r') as txt_MOD03_files,\
         open(MOD35_txt, 'r') as txt_MOD35_files :

        MOD02_paths = np.sort([path for path in txt_MOD02_files])
        MOD03_paths = np.sort([path for path in txt_MOD03_files])
        MOD35_paths = np.sort([path for path in txt_MOD35_files])

    if len(MOD02_paths) == len(MOD03_paths) and len(MOD02_paths) == len(MOD35_paths):
        return MOD02_paths, MOD03_paths, MOD35_paths
    else:
        print("files don't match up")

if __name__ == '__main__':
    import configparser
    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    #home     = config['home']['home']
    PTA      = config['current PTA']['PTA']
    PTA_path = config['PTAs'][PTA]

    MODXX_home = PTA_path + '/' + config['supporting directories']['MODXX_paths_lists']
    MOD02_txt = MODXX_home + '/MOD02_paths.txt'
    MOD03_txt = MODXX_home + '/MOD03_paths.txt'
    MOD35_txt = MODXX_home + '/MOD35_paths.txt'

    filename_MOD_02,\
    filename_MOD_03,\
    filename_MOD_35 = get_MODIS_file_paths(MOD02_txt, MOD03_txt, MOD35_txt)
    n=-1
    print(filename_MOD_02[n], filename_MOD_03[n], filename_MOD_35[n])
