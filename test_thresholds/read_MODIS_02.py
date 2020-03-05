'''
author: Javier Villegas

plotting module to visualize modis 02 product
as radiance or reflectance
'''
import numpy as np
from pyhdf.SD import SD
import pprint
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
def get_data(filename, fieldname, SD_field_rawData):
    '''
    INPUT
          filename:      string  - hdf file filepath
          fieldname:     string  - name of desired dataset
          SD_or_rawData: boolean - 0 returns SD, 1 returns field, 2 returns rawData
    RETURN SD/ raw dataset
    '''
    if SD_field_rawData==0:
        return SD(filename) #science data set
    elif SD_field_rawData==1:
        return SD(filename).select(fieldname) #field
    else:
        return SD(filename).select(fieldname).get() #raw data

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

def get_radiance_or_reflectance(data_raw, data_field, rad_or_ref):
    '''
    INPUT
          data_raw:   get_data(filename, fieldname, SD_field_rawData=2)
          data_field: get_data(filename, fieldname, SD_field_rawData=1)
          rad_or_ref: boolean - True if radiance, False if reflectance
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

    #correct raw data to get radiance/reflectance values
    #correct first band manually
    data_corrected_total = (data_raw_temp[0,:] - offset[0]) * scale_factor[0]
    #for loop to put all the bands together
    for i in range(1,num_bands):
        #corrected band
        data_corrected = (data_raw_temp[i,:] - offset[i]) * scale_factor[i]
        #aggregate bands
        data_corrected_total = np.concatenate((data_corrected_total, data_corrected), axis=0)

    #get original shape and return radiance/reflectance
    return data_corrected_total.reshape((num_bands, num_horizontal, num_vertical))

def prepare_data(filename, fieldname, rad_or_ref):
    '''
    INPUT
          filename:  string - hdf file filepath
          fieldname: string - name of desired dataset
    RETURN
          return radiance or reflectance at all bands
    '''
    data_raw   = get_data(filename, fieldname, 2)
    data_field = get_data(filename, fieldname, 1)
    rad_ref    = get_radiance_or_reflectance(data_raw, data_field, rad_or_ref)

    return rad_ref

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
    #example plot

    path = '/home/javi/MODIS_Training/MODIS data/MAIA_Science_Team_Meeting_Cases/'

    filename_MOD_02 = ['Aerosol_case/'+'MOD021KM.A2017308.0600.061.2017308193326.hdf',\
                       'Toronto_case/'+'MOD021KM.A2008159.1620.061.2017256040332.hdf']
    filename_MOD_03 = ['Aerosol_case/'+'MOD03.A2017308.0600.061.2017308125139.hdf',\
                       'Toronto_case/'+'MOD03.A2008159.1620.061.2017255201822.hdf']
    filename_MOD_35 = ['Aerosol_case/'+'MOD35_L2.A2017308.0600.061.2017308193505.hdf',\
                       'Toronto_case/'+'MOD35_L2.A2008159.1620.061.2017289200022.hdf']

    fieldnames_list  = ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB']
    rad_or_ref = False #True for radiance, False for reflectance
    plt_RGB(path + filename_MOD_02[1], fieldnames_list, rad_or_ref)
    print(get_data(filename, fieldnames_list[0], 2))

    # #debugging tools
    # file = SD('/Users/vllgsbr2/Desktop/MODIS_Training/Data/MOD021KM.A2017245.1635.061.2017258193451.hdf')
    # data = file.select('EV_500_Aggr1km_RefSB')
    # pprint.pprint(data.attributes()) #tells me scales, offsets and bands
    # pprint.pprint(file.datasets()) # shows data fields in file from SD('filename')
