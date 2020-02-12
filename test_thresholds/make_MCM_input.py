import numpy as np
from pyhdf.SD import SD
import h5py
import matplotlib.pyplot as plt

### from MOD_organize.py
def MOD021KM_read(mod02_file):

    flds = ['EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB', 'EV_Band26']

    h4f = SD(mod02_file)
    rads = []
    E_std_0 = []
    for ifld in flds:
        # Get scaled bands integers and their corresponding conversion coefficients
        sds = h4f.select(ifld)
        data_int = sds.get()

        rad_scales = sds.attributes()['radiance_scales']
        rad_offsets = sds.attributes()['radiance_offsets']
        ref_scales = sds.attributes()['reflectance_scales']
        ref_offsets = sds.attributes()['reflectance_offsets']

        if len(data_int) > 10:
            bad_rad_idx = np.where(np.array(data_int) > 32767)
            temp  = np.array((data_int - rad_offsets) * rad_scales)
            temp[bad_rad_idx] = -999
            rads.append(temp)
            E_std_0.append(np.pi*rad_scales/ref_scales)
        else:
            for iband in range(len(data_int)):
                bad_rad_idx = np.where(np.array(data_int[iband]) > 32767)
                temp  = np.array((data_int[iband] - rad_offsets[iband]) * rad_scales[iband])
                temp[bad_rad_idx] = -999
                rads.append(temp)
                E_std_0.append(np.pi*rad_scales[iband]/ref_scales[iband])


    return rads, E_std_0


### from MOD_organize.py
def MOD03_read(mod03_file):

    h4f = SD(mod03_file)
    sds = h4f.select('Latitude')
    lat = sds.get()
    sds.endaccess()

    sds = h4f.select('Longitude')
    lon = sds.get()
    sds.endaccess()

    sds = h4f.select('SolarZenith')
    sza = sds.get()
    sds.endaccess()

    sds = h4f.select('SensorZenith')
    vza = sds.get()
    sds.endaccess()

    sds = h4f.select('SolarAzimuth')
    saa = sds.get()
    sds.endaccess()

    sds = h4f.select('SensorAzimuth')
    vaa = sds.get()
    sds.endaccess()
    h4f.end()

    return lat, lon, sza, vza, saa, vaa


### from MOD_organize.py
def MOD35_read(mod35_file):

    h4f = SD(mod35_file)
    # 10-byte MODIS cloud mask quality assurance SDS (packed, refer to MOD35 UG to extract specific information)
    qa = np.array(h4f.select('Quality_Assurance')[:], dtype='int8')

    # 6-byte MODIS cloud mask SDS
    mdata = np.rollaxis(np.array(h4f.select('Cloud_Mask')[:], dtype='int8'), 0, 3)
    # last 5-byte contains the flag for individual test (packed)
    flg = mdata[:, :, 1:]
    # first 1-byte contains key information. As a result, they are unpacked and saved here.
    cld = []
    tmp = mdata[:, :, 0]%2            # 0 - not determined; 1 - determined
    cld.append(tmp)
    tmp = mdata[:, :, 0]/2%4          # 0 - cloudy; 1 - uncertain clear; 2 - probably clear; 3 - confident clear
    cld.append(tmp)
    tmp = mdata[:, :, 0]/8%2          # 0 - night; 1 - day
    cld.append(tmp)
    tmp = mdata[:, :, 0]/16%2         # 0 - sun glint; 1 - no sun glint
    cld.append(tmp)
    tmp = mdata[:, :, 0]/32%2         # 0 - snow/ice; 1 - no snow/ice
    cld.append(tmp)
    tmp = mdata[:, :, 0]/64%4         # 0 - water; 1 - coastal; 2 - desert; 3 - land
    cld.append(tmp)
    cld = np.rollaxis(np.array(cld), 0, 3)

    return cld[:, :, 1], cld[:, :, 4], cld[:, :, 5]


def make_JPL_data_from_MODIS(out_path):#MOD021KM_path, MOD03_path, MOD35_path, out_path):

    time_stamp = '2002047.1845'
    home       = '/Users/vllgsbr2/Desktop/keeling_test_files/'
    with h5py.File(home + 'LA_database.hdf5', 'r') as hf_database:
        rad_b4       = hf_database[time_stamp + '/reflectance/band_3'][()]
        rad_b5       = hf_database[time_stamp + '/reflectance/band_4'][()]
        rad_b6       = hf_database[time_stamp + '/reflectance/band_1'][()]
        rad_b9       = hf_database[time_stamp + '/reflectance/band_2'][()]
        rad_b12      = hf_database[time_stamp + '/reflectance/band_6'][()]
        rad_b13      = hf_database[time_stamp + '/reflectance/band_26'][()]
        E_std_0      = np.ones(6)
        lat          = hf_database[time_stamp + '/geolocation/lat'][()]
        lon          = hf_database[time_stamp + '/geolocation/lon'][()]
        sza          = hf_database[time_stamp + '/sunView_geometry/solarZenith'][()]
        vza          = hf_database[time_stamp + '/sunView_geometry/sensorZenith'][()]
        saa          = hf_database[time_stamp + '/sunView_geometry/solarAzimuth'][()]
        vaa          = hf_database[time_stamp + '/sunView_geometry/sensorAzimuth'][()]
        modcm        = hf_database[time_stamp + '/cloud_mask/Unobstructed_FOV_Quality_Flag'][()]
        snowice_mask = hf_database[time_stamp + '/cloud_mask/Snow_Ice_Background_Flag'][()]
        water_mask   = hf_database[time_stamp + '/cloud_mask/Land_Water_Flag'][()]

    #create hdf5 file
    hf = h5py.File(out_path, 'w')

    #define arbitrary shape for granule/orbit
    shape = (1000,1000)

    #create structure in hdf file
    ARP                = hf.create_group('Anicillary_Radiometric_Product')
    ARP_rad            = ARP.create_group('Radiance')
    ARP_RDQI           = ARP.create_group('Radiometric_Data_Quality_Indicator')
    ARP_SVG            = ARP.create_group('Sun_View_Geometry')
    ARP_earth_sun_dist = ARP.create_group('Earth_Sun_distance')
    ARP_E_std0b        = ARP.create_group('Band_Weighted_Solar_Irradiance_at_1AU')
    ARP_DOY            = ARP.create_group('Day_of_year')
    ARP_TA             = ARP.create_group('Target_Area')

    AGP     = hf.create_group('Anicillary_Geometric_Product')
    AGP_LWM = AGP.create_group('Land_Water_Mask')
    AGP_SIM = AGP.create_group('Snow_Ice_Mask')

    # #1000x1000 image patch
    # i1, i2 = 530, 1530
    # j1, j2 = 177, 1177
    #
    # #crop data to match the dimension
    # #convert
    # rad_b4  = rads[2][i1:i2, j1:j2]
    # rad_b5  = rads[3][i1:i2, j1:j2]
    # rad_b6  = rads[0][i1:i2, j1:j2]
    # rad_b9  = rads[1][i1:i2, j1:j2]
    # rad_b12 = rads[5][i1:i2, j1:j2]
    # rad_b13 = rads[7][i1:i2, j1:j2]

    RDQI_b4  = np.zeros(shape)
    RDQI_b5  = np.zeros(shape)
    RDQI_b6  = np.zeros(shape)
    RDQI_b9  = np.zeros(shape)
    RDQI_b12 = np.zeros(shape)
    RDQI_b13 = np.zeros(shape)

    SZA = sza#[i1:i2, j1:j2] / 100.
    VZA = vza#[i1:i2, j1:j2] / 100.
    SAA = saa#[i1:i2, j1:j2] / 100.
    VAA = vaa#[i1:i2, j1:j2] / 100.

    earth_sun_dist = 1

    E_std0b = E_std_0

    DOY = 6

    TA = 1

    # #0 for snow_ice, 1 for non
    # #should be consistant with MOD35 definition
    # snow_ice_mask = snowice_mask[i1:i2, j1:j2]


    # np.place(snow_ice_mask, snow_ice_mask==0, 2)
    # np.place(snow_ice_mask, snow_ice_mask==1, 0)
    # np.place(snow_ice_mask, snow_ice_mask==2, 1)

    #0 for water and 1 for land
    water_mask[water_mask > 2] = 1
    water_mask[water_mask !=1] = 0

    land_water_mask = water_mask#[i1:i2, j1:j2]
    # print(np.where(land_water_mask==0)[1].size, np.where(land_water_mask==1)[1].size)
    # plt.imshow(land_water_mask)
    # plt.colorbar()
    # plt.show()
    # print(land_water_mask)


    #assign data to groups
    ARP_rad.create_dataset('rad_band_4' , data=rad_b4 , dtype='f')
    ARP_rad.create_dataset('rad_band_5' , data=rad_b5 , dtype='f')
    ARP_rad.create_dataset('rad_band_6' , data=rad_b6 , dtype='f')
    ARP_rad.create_dataset('rad_band_9' , data=rad_b9 , dtype='f')
    ARP_rad.create_dataset('rad_band_12', data=rad_b12, dtype='f')
    ARP_rad.create_dataset('rad_band_13', data=rad_b13, dtype='f')

    ARP_RDQI.create_dataset('RDQI_band_4' , data=RDQI_b4 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_5' , data=RDQI_b5 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_6' , data=RDQI_b6 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_9' , data=RDQI_b9 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_12', data=RDQI_b12, dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_13', data=RDQI_b13, dtype='i4')

    ARP_SVG.create_dataset('solar_zenith_angle'   , data=SZA, dtype='f')
    ARP_SVG.create_dataset('viewing_zenith_angle' , data=VZA, dtype='f')
    ARP_SVG.create_dataset('solar_azimuth_angle'  , data=SAA, dtype='f')
    ARP_SVG.create_dataset('viewing_azimuth_angle', data=VAA, dtype='f')

    ARP_earth_sun_dist.create_dataset('earth_sun_dist_in_AU', data=earth_sun_dist)

    ARP_E_std0b.create_dataset('Band_Weighted_Solar_Irradiance_at_1AU', data=E_std0b, dtype='f')

    ARP_DOY.create_dataset('Day_of_year', data=DOY)

    ARP_TA.create_dataset('Target_Area', data=TA)

    AGP_LWM.create_dataset('Land_Water_Mask', data=land_water_mask, dtype='i4', compression='gzip')
    AGP_SIM.create_dataset('Snow_Ice_Mask', data=snow_ice_mask, dtype='i4', compression='gzip')

    hf.close()

    return lat, lon, modcm

if __name__ == '__main__':
    #this makes the JPL data file to read into the MCM
    out_path      = './test_JPL_MODIS_data_.HDF5'
    make_JPL_data_from_MODIS(out_path)
