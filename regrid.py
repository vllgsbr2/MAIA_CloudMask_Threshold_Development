from read_MODIS_02 import get_data, get_radiance_or_reflectance, prepare_data
from read_MODIS_03 import get_lat, get_lon
from read_MODIS_35 import *
import pytaf#from PTA_Subset import crop_PTA
#from multicore_processing import multi_core_crop
import h5py
import sys
import os
import time
from pyhdf.SD import SD


def regrid_MODIS_2_MAIA(source_lat, source_lon, target_lat, target_lon, source_data):
    '''
    Objective:
        take source geolocation and source data, and regrid it to a target
        geolocation using nearest nieghbor.
    Arguments:
        source_lat, source_lon {2D narrays} -- lat/lon of data to be regridded
        target_lat, target_lon {2D narrays} -- lat/lon of refrence grid
        source_data {2D narray} -- data that will be regridded
    Returns:
        target_data {2D narray} -- returns regridded source data, such that it
                                   matches up with the target lat/lon
    '''
    #radius in meters to search around pixel for a neighbor
    max_radius = 5556.
    target_data = pytaf.resample_n(source_lat, source_lon, target_lat,\
                                   target_lon, source_data, max_radius)
    return target_data

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import h5py 
    import copy 
    #home       = '/data/keeling/a/vllgsbr2/c/LA_test_case_data/'
    #file_mod03 = home + 'MOD03.A2017246.1855.061.2017257170030.hdf'
    #file_mod02 = home + 'MOD021KM.A2017246.1855.061.2017258202757.hdf'
    home       = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data'   
    file_mod03 = home + '/MOD_03/MOD03.A2002002.1920.061.2017179152611.hdf'
    file_mod02 = home + '/MOD_02/MOD021KM.A2002002.1920.061.2017179161255.hdf'

    file_MAIA  = '/data/keeling/a/vllgsbr2/c/LA_PTA_MAIA.hdf5' 
    file_MAIA  = h5py.File(file_MAIA, 'r') 

    #load in lat/lon from modis granule
    s_lat = get_lat(file_mod03).astype(np.float64)
    s_lon = get_lon(file_mod03).astype(np.float64)
    t_lat = file_MAIA['lat'][()].astype(np.float64)
    t_lon = file_MAIA['lon'][()].astype(np.float64)
    
    #load in data to be regridded from modis granule
    red_band = prepare_data(file_mod02, 'EV_250_Aggr1km_RefSB', False)
    s_data   = red_band[0,:,:].astype(np.float64)
    

    #do something illegal
    #make copies for regridding routine to modify
    target_lat_row = np.copy(t_lat)
    target_lon_row = np.copy(t_lon)
    target_lat_col = np.copy(t_lat)
    target_lon_col = np.copy(t_lon)
    #new indices to regrid, use that as numpy where if you will
    nx, ny = (2030, 1354)
    rows = np.arange(2030)
    cols = np.arange(1354)
    col_mesh, row_mesh = np.meshgrid(cols, rows)

    import time

    for i in range(5):

        start=time.time()

        regrid_row_idx =regrid_MODIS_2_MAIA(np.copy(s_lat), np.copy(s_lon), np.copy(t_lat), np.copy(t_lon),\
                                         np.copy(row_mesh).astype(np.float64)).astype(np.int)

        regrid_col_idx = regrid_MODIS_2_MAIA(np.copy(s_lat), np.copy(s_lon), np.copy(t_lat), np.copy(t_lon),\
                                         np.copy(col_mesh).astype(np.float64)).astype(np.int)
    
        print(-start+time.time())
        #print(regrid_row_idx.shape, regrid_col_idx.shape)

        #get the regridded data
        target_data = s_data[regrid_row_idx, regrid_col_idx]#regrid_MODIS_2_MAIA(s_lat, s_lon, t_lat, t_lon, s_data)
        s_data[regrid_row_idx, regrid_col_idx] += 0.5

        #plot the MAIA data next to modis data to see if result makes sense
        #print(np.shape(s_data), np.shape(target_data))
        f, ax = plt.subplots(ncols=2)
        #ax[0].imshow(regrid_col_idx)
        #ax[1].imshow(regrid_row_idx)    
        ax[0].imshow(s_data)
        ax[1].imshow(target_data)
        #f.savefig('./test_pytaf.png', dpi=300)
        plt.show()
