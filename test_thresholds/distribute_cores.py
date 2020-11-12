import numpy as np

def distribute_processes(size, num_processes):

  #must calculate cores for each cpu every time; or dedicate one core to this
  #it is better to calculate each time becuase it is very fast. No need to waste
  #a core on it
  start, stop = [], []
  for rank in range(size):
    #distribute processes over cores allocated until no more processes are left over
    processes_per_core  = num_processes // (size)
    left_over_processes = num_processes % (size)
    #number of processes each core gets before adding left over
    processes_per_core_list = (np.ones((size))*processes_per_core).astype(np.int)

    while left_over_processes > 0:
      #redeclare amount of cores available for use
      num_cores = np.copy(size)
      #add one process to every core until left_over_processes run out
      while num_cores-1 >= 0:
        #add a process to a core one at a time
        processes_per_core_list[num_cores-1] += 1
        #goto next core
        num_cores -= 1
        #eliminate process added to core from left over count
        left_over_processes -= 1
        #if we run out of processes to distribute, stop
        if left_over_processes == 0:
          break

    #declare the start and stop values to use as indices for files
    #processed by each core
    start.append(processes_per_core_list[:rank].sum())
    stop.append(start[rank] + processes_per_core_list[rank])

    print(rank, start[rank], stop[rank], stop[rank]-start[rank])

  return start, stop

if __name__=='__main__':
    #example sudo code
    #import mpi4py and set up the for loop for to cycle through the ranks
    #size refers to the number of allocated cores in the mpi4py convention
    size          = 200
    #num process simply refers to howmany processes must be completed by the cores
    num_processes = 9873

    #start stop are lists of indices that refer to the subset of processes each core should handle
    start, stop = distribute_processes(size, num_processes)
    #files_to_process = ['file1', 'file_2', 'file_3','file_n']
    #f2p = files_to_process[start[rank]:stop[rank]]
    #then call all your functions using subset of data calculated for each core 
