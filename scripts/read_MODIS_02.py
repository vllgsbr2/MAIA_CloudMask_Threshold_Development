'''
author: Javier Villegas

plotting module to visualize modis 02 product
as radiance or reflectance
'''
import numpy as np
from pyhdf.SD import SD
#import h5py
#import pprint
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

def get_earth_sun_dist(filename_MOD_02):
    file_ = SD(filename_MOD_02)
    earthsundist = getattr(file_, 'Earth-Sun Distance')
    file_.end()

    return earthsundist

def get_data(filename, fieldname, SD_field_rawData, return_hdf=False):
    '''
    INPUT
          filename:      string  - hdf file filepath
          fieldname:     string  - name of desired dataset
          SD_or_rawData: boolean - 0 returns SD, 1 returns field, 2 returns rawData
    RETURN SD/ raw dataset
    '''
    hdf_file = SD(filename)
    if SD_field_rawData==0:
        data = hdf_file #science data set
    elif SD_field_rawData==1:
        data = hdf_file.select(fieldname) #field
    else:
        data = hdf_file.select(fieldname).get() #raw data

    # hdf_file.end()
    if return_hdf:
        return data, hdf_file

    return data

def get_scale_and_offset(data_field, rad_or_ref):
    '''
    INPUT
          data:       numpy float array - get_data(filename,fieldname,SD_or_rawData=1)
          rad_or_ref: boolean           - True if radaince or False if reflectance offsets/scale desired
    RETURN
          2 numpy float arrays, scale factor & offset of size=number of bands
          for the chosen field
    '''

    if rad_or_ref:
        offset_name = 'radiance_offsets'
        scale_name  = 'radiance_scales'
    else:
        offset_name = 'reflectance_offsets'
        scale_name  = 'reflectance_scales'

    #grab scale and offset
    scale_factor = data_field.attributes()[scale_name]
    offset = data_field.attributes()[offset_name]

    return scale_factor, offset

def get_radiance_or_reflectance(data_raw, data_field, rad_or_ref, scale_factor=True):
    '''
    INPUT
          data_raw:   get_data(filename, fieldname, SD_field_rawData=2)
          data_field: get_data(filename, fieldname, SD_field_rawData=1)
          rad_or_ref: boolean - True if radiance, False if reflectance
          scale_factor: boolean - return this as well if True
    RETURN
          radiance: numpy float array - shape=(number of bands, horizontal, vertical)
    '''
    #get dimensions of raw data
    num_bands = np.ma.size(data_raw, axis=0)
    num_horizontal = np.ma.size(data_raw, axis=1)
    num_vertical = np.ma.size(data_raw, axis=2)

    #reshape raw data to perform scale and offset correction
    data_raw_temp = np.reshape(data_raw,(num_bands, num_horizontal * num_vertical))
    scale_factor, offset = get_scale_and_offset(data_field, rad_or_ref)

    #fill values found in data
    detector_saturated = 65533
    detector_dead      = 65531
    missing_data       = 65535
    max_DN             = 32767
    min_DN             = 0

    #save indices of where bad values occured occured
    detector_saturated_idx = np.where(data_raw_temp == detector_saturated)
    detector_dead_idx      = np.where(data_raw_temp == detector_dead)
    missing_data_idx       = np.where(data_raw_temp == missing_data)
    over_DN_max_idx        = np.where(data_raw_temp >  max_DN)
    below_min_DN_idx       = np.where(data_raw_temp <  min_DN)

    #correct raw data to get radiance/reflectance values
    #correct first band manually
    data_corrected_total = (data_raw_temp[0,:] - offset[0]) * scale_factor[0]
    #for loop to put all the bands together
    for i in range(1,num_bands):
        #corrected band
        data_corrected = (data_raw_temp[i,:] - offset[i]) * scale_factor[i]
        #reinput fill vals to be queried later
        data_corrected[detector_saturated_idx[i]] = detector_saturated
        data_corrected[detector_dead_idx[i]]      = detector_dead
        data_corrected[missing_data_idx[i]]       = missing_data
        data_corrected[over_DN_max_idx[i]]        = over_DN_max
        data_corrected[below_min_DN_idx[i]]       = below_min_DN
        #aggregate bands
        data_corrected_total = np.concatenate((data_corrected_total, data_corrected), axis=0)

    #get original shape and return radiance/reflectance
    if not scale_factor:
        return data_corrected_total.reshape((num_bands, num_horizontal, num_vertical))
    else:
        scale_factor_rad, offset = get_scale_and_offset(data_field, True)
        scale_factor_ref, offset = get_scale_and_offset(data_field, False)
        return data_corrected_total.reshape((num_bands, num_horizontal, num_vertical)),\
               scale_factor_rad, scale_factor_ref

def prepare_data(filename, fieldname, rad_or_ref):
    '''
    INPUT
          filename:  string - hdf file filepath
          fieldname: string - name of desired dataset
    RETURN
          return radiance or reflectance at all bands
    '''
    data_raw, hdf_file1   = get_data(filename, fieldname, 2, True)
    data_field, hdf_file2 = get_data(filename, fieldname, 1, True)
    rad_ref, scale_factor_rad, scale_factor_ref = get_radiance_or_reflectance(data_raw, data_field, rad_or_ref)

    hdf_file1.end()
    hdf_file2.end()

    return rad_ref, scale_factor_rad, scale_factor_ref

def plt_RGB(filename, fieldnames_list, rad_or_ref, plot=True):
    '''
    INPUT
          filename:        - string     , filepath to file
          fieldnames_list: - string list, contains 500m res and 250m reshape
                                          such that bands 1,4,3 for RGB
                                          i.e. 'EV_500_Aggr1km_RefSB'
    RETURN
          plots RGB picture of MODIS 02 product data
    '''


    #make channels for RGB photo (index 01234 -> band 34567)
    image_blue  = prepare_data(filename, fieldnames_list[0],rad_or_ref)[0,:,:] #band 3 from 500 meter res
    image_green = prepare_data(filename, fieldnames_list[0],rad_or_ref)[1,:,:] #band 4 from 500 meter res
    image_red   = prepare_data(filename, fieldnames_list[1],rad_or_ref)[0,:,:] #band 1 from 250 meter res

    #force reflectance values to max out at 1.0/ normalize radiance
    if not rad_or_ref:
        np.place(image_red, image_red>1.0, 1.0) #2d image array, condition, value
        np.place(image_blue, image_blue>1.0, 1.0)
        np.place(image_green, image_green>1.0, 1.0)
        image_RGB = np.dstack([image_red, image_green, image_blue])

    else:
        #use astropy to normalize radiance values to usable pixel brightness
        from astropy.visualization import make_lupton_rgb
        image_RGB = make_lupton_rgb(image_red, image_green, image_blue, stretch=0.5)


    #plot or return image
    if plot:
        plt.imshow(image_RGB)
        plt.show()
    else:
        return image_RGB

if __name__ == '__main__':
    pass
    # ##example plot
    # filename   = '/u/sciteam/villegas/MAIA_Threshold_Development/test_data/MOD021KM.A2017118.1715.061.2017314055816.hdf'
    # fieldnames_list  = ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB']
    # rad_or_ref = True #True for radiance, False for reflectance
    # plt_RGB(filename, fieldnames_list, rad_or_ref)
    # print(get_data(filename, fieldnames_list[0], 2))

    #plot images from LA database to check the data
    #fieldnames_list = ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB', 'EV_Band26', 'EV_1KM_RefSB']
    #rad_or_ref      = False
    #home            = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/MOD_02/'
    #filename        = home + 'MOD021KM.A2002051.1820.061.2017180020820.hdf'
    #ref_band_26, scale_factor_rad, scale_factor_ref = prepare_data(filename, fieldnames_list[3], rad_or_ref)
    #plt.imshow(ref_band_26[14,:,:], cmap='Greys_r', vmax=.3)
    #plt.show()
    #plt_RGB(filename, fieldnames_list, rad_or_ref)
    #path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development/test_thresholds/test_JPL_MODIS_data_.HDF5'
    #import h5py
    #JPL_file = h5py.File(path, 'r')

    #retrieve radiances
    #rad_band_4  = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_13']
    #plt.imshow(rad_band_4, cmap='Greys_r')

    #plt.show()
    # #debugging tools
    #file = SD('/Users/vllgsbr2/Desktop/MODIS_Training/Data/MOD021KM.A2017245.1635.061.2017258193451.hdf')
    #data = file.select('EV_1KM_Emissive')
    #pprint.pprint(data.attributes()) #tells me scales, offsets and bands
    #pprint.pprint(file.datasets()) shows data fields in file from SD('filename')
