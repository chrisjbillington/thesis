# To be run with mpi like so:
#     mpirun -n <N_PROCESSES> python mpi_test.py

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

with h5py.File('mpi_test.h5', 'w', driver='mpio', comm=comm) as f:
    dset = f.create_dataset('test', (SIZE,), dtype=int)
    dset[RANK] = RANK**2
