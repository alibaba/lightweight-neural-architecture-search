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


def __acquire_gpu__(acquired_gpu_list_filename, gpu_id_list, num_acq_gpus):
    mkfilepath(acquired_gpu_list_filename)
    gpu_id_set = set(gpu_id_list)
    new_acquire_gpu_id_list = None
    if not os.path.isfile(acquired_gpu_list_filename):
        with open(acquired_gpu_list_filename, 'w') as fid:
            new_gpu_id_list = list(gpu_id_set)
            new_acquire_gpu_id_list = new_gpu_id_list[0:num_acq_gpus]
            new_acquire_gpu_id_str = ','.join(new_acquire_gpu_id_list)
            fid.write(new_acquire_gpu_id_str + ',')
            print(new_acquire_gpu_id_str)
        pass  # end with
    else:
        # already has one file, first remove acquired gpus from our gpu_id_set
        with open(acquired_gpu_list_filename, 'r') as fid:
            the_lines = fid.readlines()
        pass

        the_lines = [s.strip() for s in the_lines] # remove \n        

        acquired_gpu_id_str = ','.join(the_lines)
        acquired_gpu_id_list = acquired_gpu_id_str.split(',')
        acquired_gpu_id_list = [x for x in acquired_gpu_id_list if len(x) > 0] # remove empty
        acquired_gpu_id_set = set(acquired_gpu_id_list)
        unused_gpu_id_set = gpu_id_set - acquired_gpu_id_set

        # second step, acquire remaining gpus and append to record files.
        with open(acquired_gpu_list_filename, 'a') as fid:
            new_gpu_id_list = list(unused_gpu_id_set)
            new_acquire_gpu_id_list = new_gpu_id_list[0:num_acq_gpus]
            new_acquire_gpu_id_str = ','.join(new_acquire_gpu_id_list)
            fid.write(new_acquire_gpu_id_str + ',')
            print(new_acquire_gpu_id_str)
        pass  # end with

    pass

    return new_acquire_gpu_id_list


pass


def acquire_gpu(gpu_id_list, num_acq_gpus):
    lock_filename = './acquired_gpu_list.lock'
    acquired_gpu_list_filename = './acquired_gpu_list.txt'
    mkfilepath(lock_filename)
    mkfilepath(acquired_gpu_list_filename)

    if gpu_id_list == 'auto':
        raise RuntimeError('Not implemented!')
    else:
        pass
    pass


    lock = FileLock(lock_filename)

    is_acquire_success = False
    new_acquire_gpu_id_list = None
    for retry_count in range(3):
        try:
            with lock.acquire(timeout=10):
                new_acquire_gpu_id_list = __acquire_gpu__(acquired_gpu_list_filename, gpu_id_list, num_acq_gpus)
                is_acquire_success = True
            pass  # end with
        except Timeout:
            if os.path.isfile(lock_filename):
                os.remove(lock_filename)
            pass
        pass  # end try

        if is_acquire_success:
            break
        pass  # end if
    pass  # end for

    if not is_acquire_success:
        raise RuntimeError('Cannot acquire any free GPU')

    return new_acquire_gpu_id_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id_list', type=str, help='string of available gpu ID list')
    parser.add_argument(
        '--num_acq_gpus', type=int, help='number of gpus to acquire.')

    opt, unknown_args = parser.parse_known_args(sys.argv)

    gpu_id_list = opt.gpu_id_list.split(',')

    acquire_gpu(gpu_id_list=gpu_id_list, num_acq_gpus=opt.num_acq_gpus)

