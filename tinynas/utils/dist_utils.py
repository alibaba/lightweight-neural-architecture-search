import functools
from typing import Callable, List, Optional, Tuple

def get_dist_info() -> Tuple[int, int]:

    try:
        import mpi4py.MPI as MPI 
        mpi_comm = MPI.COMM_WORLD
        rank = mpi_comm.Get_rank()
        world_size = mpi_comm.Get_size()
    except ImportError:
        rank = 0
        world_size = 1
    return rank, world_size

def is_master():
    rank, _ = get_dist_info()
    return rank == 0

def master_only(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def get_mpi_comm():
    try:
       import mpi4py.MPI as MPI
       mpi_comm = MPI.COMM_WORLD
    except ImportError:
       mpi_comm = None
    return mpi_comm

def worker_only(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank > 0:
            return func(*args, **kwargs)

    return wrapper
