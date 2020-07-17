import numpy as np
from pyhdf.SD import SD
import h5py
import matplotlib.pyplot as plt
import os

def make_JPL_data_from_MODIS(database_file, output_path):

    with h5py.File(database_file, 'r') as hf_database:
        keys = list(hf_database.keys())

        if len(keys) > 0:
            for time_stamp in keys:
                rad_b4             = hf_database[time_stamp + '/radiance/band_3'][()]
                rad_b5             = hf_database[time_stamp + '/radiance/band_4'][()]
                rad_b6             = hf_database[time_stamp + '/radiance/band_1'][()]
                rad_b9             = hf_database[time_stamp + '/radiance/band_2'][()]
                rad_b12            = hf_database[time_stamp + '/radiance/band_6'][()]
                rad_b13            = hf_database[time_stamp + '/radiance/band_26'][()]
                E_std0b            = hf_database[time_stamp + '/band_weighted_solar_irradiance'][()]
                lat                = hf_database[time_stamp + '/geolocation/lat'][()]
                lon                = hf_database[time_stamp + '/geolocation/lon'][()]
                SZA                = hf_database[time_stamp + '/sunView_geometry/solarZenith'][()]
                VZA                = hf_database[time_stamp + '/sunView_geometry/sensorZenith'][()]
                SAA                = hf_database[time_stamp + '/sunView_geometry/solarAzimuth'][()]
                VAA                = hf_database[time_stamp + '/sunView_geometry/sensorAzimuth'][()]
                modcm              = hf_database[time_stamp + '/cloud_mask/Unobstructed_FOV_Quality_Flag'][()]
                snow_ice_mask      = hf_database[time_stamp + '/cloud_mask/Snow_Ice_Background_Flag'][()]
                water_mask         = hf_database[time_stamp + '/cloud_mask/Land_Water_Flag'][()]
                earth_sun_distance = hf_database[time_stamp + '/earth_sun_distance'][()]
                MOD03_LandSeaMask     = hf_database[time_stamp + '/MOD03_LandSeaMask'][()]

                #create hdf5 file
                hf = h5py.File(output_path + 'test_JPL_data_{}.h5'.format(time_stamp), 'w')

                #define arbitrary shape for granule/orbit
                shape = rad_b4.shape #(1000,1000)

                #add cloud mask for later purposes
                hf.create_dataset('MOD35_cloud_mask', data=modcm, compression='gzip')

                #create structure in hdf file
                ARP                = hf.create_group('Anicillary_Radiometric_Product')
                ARP_rad            = ARP.create_group('Radiance')
                ARP_RDQI           = ARP.create_group('Radiometric_Data_Quality_Indicator')
                ARP_SVG            = ARP.create_group('Sun_View_Geometry')
                ARP_earth_sun_dist = ARP.create_group('Earth_Sun_distance')
                ARP_E_std0b        = ARP.create_group('Band_Weighted_Solar_Irradiance_at_1AU')
                ARP_DOY            = ARP.create_group('Day_of_year')
                ARP_TA             = ARP.create_group('Target_Area')
                ARP_ESD            = ARP.create_group('Earth_Sun_Distance')
                ARP_LSM            = ARP.create_group('MOD03_LandSeaMask')

                AGP     = hf.create_group('Anicillary_Geometric_Product')
                AGP_LWM = AGP.create_group('Land_Water_Mask')
                AGP_SIM = AGP.create_group('Snow_Ice_Mask')

                RDQI_b4  = np.zeros(shape)
                RDQI_b5  = np.zeros(shape)
                RDQI_b6  = np.zeros(shape)
                RDQI_b9  = np.zeros(shape)
                RDQI_b12 = np.zeros(shape)
                RDQI_b13 = np.zeros(shape)

                RDQI_b4[rad_b4==-999]   = 3
                RDQI_b5[rad_b5==-999]   = 3
                RDQI_b6[rad_b6==-999]   = 3
                RDQI_b9[rad_b9==-999]   = 3
                RDQI_b12[rad_b12==-999] = 3
                RDQI_b13[rad_b13==-999] = 3

                DOY = int(time_stamp[4:7])

                TA = 1

                #0 for water and 1 for land
                water_mask[water_mask >= 2] = 1
                water_mask[water_mask !=1] = 0

                land_water_mask = water_mask

                #assign data to groups
                ARP_rad.create_dataset('rad_band_4' , data=rad_b4 , dtype='f', compression='gzip')
                ARP_rad.create_dataset('rad_band_5' , data=rad_b5 , dtype='f', compression='gzip')
                ARP_rad.create_dataset('rad_band_6' , data=rad_b6 , dtype='f', compression='gzip')
                ARP_rad.create_dataset('rad_band_9' , data=rad_b9 , dtype='f', compression='gzip')
                ARP_rad.create_dataset('rad_band_12', data=rad_b12, dtype='f', compression='gzip')
                ARP_rad.create_dataset('rad_band_13', data=rad_b13, dtype='f', compression='gzip')

                ARP_RDQI.create_dataset('RDQI_band_4' , data=RDQI_b4 , dtype='i4', compression='gzip')
                ARP_RDQI.create_dataset('RDQI_band_5' , data=RDQI_b5 , dtype='i4', compression='gzip')
                ARP_RDQI.create_dataset('RDQI_band_6' , data=RDQI_b6 , dtype='i4', compression='gzip')
                ARP_RDQI.create_dataset('RDQI_band_9' , data=RDQI_b9 , dtype='i4',  compression='gzip')
                ARP_RDQI.create_dataset('RDQI_band_12', data=RDQI_b12, dtype='i4', compression='gzip')
                ARP_RDQI.create_dataset('RDQI_band_13', data=RDQI_b13, dtype='i4', compression='gzip')

                ARP_SVG.create_dataset('solar_zenith_angle'   , data=SZA, dtype='f', compression='gzip')
                ARP_SVG.create_dataset('viewing_zenith_angle' , data=VZA, dtype='f', compression='gzip')
                ARP_SVG.create_dataset('solar_azimuth_angle'  , data=SAA, dtype='f', compression='gzip')
                ARP_SVG.create_dataset('viewing_azimuth_angle', data=VAA, dtype='f', compression='gzip')

                ARP_ESD.create_dataset('earth_sun_dist_in_AU', data=earth_sun_distance)

                ARP_E_std0b.create_dataset('Band_Weighted_Solar_Irradiance_at_1AU', data=E_std0b, dtype='f')

                ARP_DOY.create_dataset('Day_of_year', data=DOY)

                ARP_TA.create_dataset('Target_Area', data=TA)

                ARP_LSM.create_dataset('MOD03_LandSeaMask', data=MOD03_LandSeaMask)

                AGP_LWM.create_dataset('Land_Water_Mask', data=land_water_mask, dtype='i4', compression='gzip')
                AGP_SIM.create_dataset('Snow_Ice_Mask', data=snow_ice_mask, dtype='i4', compression='gzip')

                hf.close()

                print(time_stamp)

if __name__ == '__main__':

    import mpi4py.MPI as MPI
    import configparser

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
            config = configparser.ConfigParser()
            config.read(config_home_path+'/test_config.txt')

            PTA          = config['current PTA']['PTA']
            PTA_path     = config['PTAs'][PTA]

            database_path  = '{}/{}/'.format(PTA_path, config['supporting directories']['Database'])
            database_files = [database_path + x for x in os.listdir(database_path)]
            output_path    = '{}/{}/'.format(PTA_path, config['supporting directories']['MCM_Input'])

            #this makes the JPL data file to read into the MCM
            make_JPL_data_from_MODIS(database_files[r], output_path)
