from regrid import regrid_MODIS_2_MAIA
from netCDF4 import Dataset
import h5py
import os
import numpy as np
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for r in range(size):
    if rank==r:

        home   = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data'
        sfcID_x = ['{}/{}/{}'.format(home, 'try2_database/LA_surface_types_old_grid', x)\
                   for x in os.listdir(home+'/try2_database/LA_surface_types_old_grid')][r]

        LA_grid_file = '{}/{}/{}'.format(home, 'PTA_lat_lon_grids', 'Grids_USA_LosAngeles.h5')
        with h5py.File(LA_grid_file, 'r') as hf_LA_grid:
            target_lat = hf_LA_grid['Geolocation/Latitude'][()].astype(np.float64)
            target_lon = hf_LA_grid['Geolocation/Longitude'][()].astype(np.float64)

        with Dataset(sfcID_x, 'r') as nc_surfaceID:
            source_lat  = nc_surfaceID.variables['Latitude'][:].astype(np.float64)
            source_lon  = nc_surfaceID.variables['Longitude'][:].astype(np.float64)
            source_data = nc_surfaceID.variables['surface_ID'][:].astype(np.float64)

            print('regridding {}'.format(sfcID_x[-19:]))
            sfcID_regridded = regrid_MODIS_2_MAIA(np.copy(source_lat), np.copy(source_lon),\
                                                  np.copy(target_lat), np.copy(target_lon),\
                                                  np.copy(source_data)).astype(np.int)
            print('saving new {}'.format(sfcID_x[-19:]))
            #reassign regridded data to new file
            regrid_sfcID_file = '{}/{}/{}'.format(home, 'try2_database/LA_surface_types', sfcID_x[-19:])
            with Dataset(regrid_sfcID_file, 'w') as nc_regrid_sfcID_x:
                shape = sfcID_regridded.shape
                grid = nc_regrid_sfcID_x.createDimension('grid', shape)

                nc_regrid_sfcID_x.createVariable('Latitude'  , 'f4', 'grid')
                nc_regrid_sfcID_x.createVariable('Longitude' , 'f4', 'grid')
                nc_regrid_sfcID_x.createVariable('surface_ID', 'f4', 'grid')

                nc_regrid_sfcID_x.variables['Latitude'][:]   = target_lat
                nc_regrid_sfcID_x.variables['Longitude'][:]  = target_lon
                nc_regrid_sfcID_x.variables['surface_ID'][:] = sfcID_regridded

        print('successfully regridded and saved {}'.format(sfcID_x[-19:]))
