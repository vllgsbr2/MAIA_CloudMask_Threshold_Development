import h5py
import numpy as np
import mpi4py.MPI as MPI
import tables
tables.file._open_files.close_all()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# for each granule (1000*1000, 9, 1, 7) -> (pixels, OLP bins, CM, observables)

for r in range(size):
    if rank==r:
        #first open the OLP, observables and the PTA database and make intermediate database
        with h5py.File('','r') as hf_OLP,\
             h5py.File('','r') as hf_observables,\
             h5py.File('','r') as hf_PTA_database,\
             h5py.File('','w') as hf_intermediate:

            time_stamps = list(hf_PTA_database.keys())

            observable_names = ['WI', 'NDVI', 'NDSI', 'visRef', 'nirRef', 'SVI', 'cirrus']

            for time in time_stamps:

                try:
                    group_time = hf_intermediate.create_group(time)

                    group_OLP = group_time.create_group('observable_level_parameter')
                    group_obs = group_time.create_group('observables')
                    group_cm  = group_time.create_group('cloud_mask')

                except:
                    pass

                observables = {}
                for obs in observable_names:
                    observables[obs] = hf_observables[time+'/'+obs]

                #(1000,1000,9)
                OLP         = hf_OLP[time+'/observable_level_parameter']
                cloud_mask  = hf_PTA_database[time+'/cloud_mask/Unobstructed_FOV_Quality_Flag']

                #save into hf_intermediate
                try:
                    for obs in observable_names:
                        group_obs.create_dataset(obs ,data=observables[obs] ,compression='gzip)

                    group_OLP.create_dataset('observable_level_parameter',data=OLP,compression='gzip)
                    group_cm.create_dataset('cloud_mask',data=cloud_mask ,compression='gzip)

                except:

                    for obs in observable_names:
                        hf_intermediate[time+'/observables/'+obs] = observables[obs]

                    hf_intermediate[time+'/observable_level_parameter/observable_level_parameter'] = OLP
                    hf_intermediate[time+'/cloud_mask'] = cloud_mask
