from read_MODIS_02 import *
import numpy as np
from pyhdf.SD import *

#print all attributes 
#first do swath meta data
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

if __name__ == '__main__':
    #example plot
    path = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/MOD_02/'
    filename_MOD_02 = path + 'MOD021KM.A2010354.1810.061.2017256205718.hdf'
    
    #debugging tools
    file_ = SD(filename_MOD_02)
    #data = file_.select('Swath Attributes')
    ###pprint.pprint(file_.attributes())
    #pprint.pprint(file_.attr('Earth-Sun Distance'))#attributes()) #tells me scales, offsets and bands
    #pprint.pprint(file.datasets()) # shows data fields in file from SD('filename')
    #esd = file_.attr('Earth-Sun Distance')
    #pprint(esd.get())
    #pprint.pprint(getattr(file_, 'Earth-Sun Distance'))

    esd = getattr(file_, 'Earth-Sun Distance')
    file_.end()
    print(esd)
