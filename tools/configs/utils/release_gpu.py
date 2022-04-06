# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from filelock import Timeout, FileLock
import distutils


def mkfilepath(filename):
    filename = os.path.expanduser(filename)
    distutils.dir_util.mkpath(os.path.dirname(filename))


def mkdir(dirname):
    dirname = os.path.expanduser(dirname)
    distutils.dir_util.mkpath(dirname)


def __release_gpu__(acquired_gpu_list_filename, to_release_gpu_id_set):
    if not os.path.isfile(acquired_gpu_list_filename):
        return
    pass

    with open(acquired_gpu_list_filename, 'r') as fid:
        the_lines = fid.readlines()
    pass

    the_lines = [s.strip() for s in the_lines] # remove \n
    the_lines = ','.join(the_lines)
    acquired_gpu_id_list = the_lines.split(',')
    acquired_gpu_id_list = [x for x in acquired_gpu_id_list if len(x)>0] # remove empty
    acquired_gpu_id_set = set(acquired_gpu_id_list)

    used_gpu_id_set = acquired_gpu_id_set - to_release_gpu_id_set
    used_gpu_id_list = list(used_gpu_id_set)

    with open(acquired_gpu_list_filename, 'w') as fid:
        fid.write(','.join(used_gpu_id_list) + ',')
    pass
    
pass


def release_gpu(release_gpu_id_list):
    lock_filename = './acquired_gpu_list.lock'
    acquired_gpu_list_filename = './acquired_gpu_list.txt'
    mkfilepath(lock_filename)
    mkfilepath(acquired_gpu_list_filename)

    to_release_gpu_id_set = set(release_gpu_id_list)
    lock = FileLock(lock_filename)

    is_success = False
    for retry_count in range(3):
        try:
            with lock.acquire(timeout=10):
                __release_gpu__(acquired_gpu_list_filename, to_release_gpu_id_set)
                is_success = True
            pass  # end with
        except Timeout:
            if os.path.isfile(lock_filename):
                os.remove(lock_filename)
            pass
        pass  # end try

        if is_success:
            break
        pass  # end if
    pass  # end for

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--release_gpu_id_list', type=str, help='string of gpu ID to release')

    opt, unknown_args = parser.parse_known_args(sys.argv)
    release_gpu_id_list = opt.release_gpu_id_list.split(',')
    release_gpu(release_gpu_id_list=release_gpu_id_list)



