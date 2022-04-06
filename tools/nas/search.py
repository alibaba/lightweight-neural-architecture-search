# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os
import sys
import pdb
import time
import copy
import random
import warnings
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import (get_root_logger, load_py_module_from_path, 
                AutoGPU, load_pyobj, save_pyobj, DictAction)
from nas.builder import BuildNAS
from nas.evolutions import Population 


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def __check_block_structure_info_list_valid__(block_structure_info_list, cfg):
    if len(block_structure_info_list) < 1:
        return False

    # first block must be ConvKXBNRELU with in_channels=3
    #if block_structure_info_list[0]['class'] != 'ConvKXBNRELU' or block_structure_info_list[0]['in'] != 3:
        #return False

    # check how many conv layers and stages
    layers = 0
    num_stages = 0
    for block_structure_info in block_structure_info_list:
        stride = block_structure_info['s']
        if stride == 2:
            num_stages += 1
        else:
            assert stride == 1

        if "L" not in block_structure_info.keys():
            layers += 1
        elif block_structure_info['L'] == "SuperResConvKXKX":
            layers += block_structure_info['L']*2
        else:
            layers += block_structure_info['L']*3

    if cfg.budget_stages is not None and num_stages > cfg.budget_stages:
        return False
    
    if cfg.budget_layers is not None and layers > cfg.budget_layers:
        return False

    return True


def adjust_structures_inplace(block_structure_info_list, cfg):

    # adjust channels
    last_channels = None
    for i, block_structure_info in enumerate(block_structure_info_list):
        if last_channels is None:
            last_channels = block_structure_info['out']
            continue
        else:
            block_structure_info_list[i]['in'] = last_channels
            last_channels = block_structure_info['out']

    # adjust kernel size <= feature map / 1.5
    resolution = cfg.budget_image_size
    for i, block_structure_info in enumerate(block_structure_info_list):
        stride = block_structure_info['s']
        kernel_size = block_structure_info['k']

        while kernel_size * 1.5 > resolution:
            kernel_size -= 2

        block_structure_info['k'] = kernel_size

        resolution /= stride

    return block_structure_info_list


def get_new_random_structure_info(block_structure_info_list, mutate_function, cfg, \
                                minor_mutation=False):
    block_structure_info_list = copy.deepcopy(block_structure_info_list)

    for mutate_count in range(cfg.space_block_num):
        is_valid = False

        for idx in range(len(block_structure_info_list)):
            random_id = random.randint(0, len(block_structure_info_list) - 1)

            if cfg.space_exclude_stem:
                while random_id == 0:
                    random_id = random.randint(0, len(block_structure_info_list) - 1)

            mutated_block_list = mutate_function(random_id, block_structure_info_list, \
                cfg.budget_layers, minor_mutation=minor_mutation)

            if mutated_block_list == False:
                continue

            new_block_structure_info_list = []
            for block_id in range(len(block_structure_info_list)):
                if block_id != random_id:
                    new_block_structure_info_list.append(block_structure_info_list[block_id])
                else:
                    if mutated_block_list is None:
                        pass
                    else:
                        for mutated_block in mutated_block_list:
                            new_block_structure_info_list.append(mutated_block)
                        pass
                    pass  # end if
                pass  # end if

            adjust_structures_inplace(new_block_structure_info_list, cfg)
            # check valid
            is_valid = __check_block_structure_info_list_valid__(new_block_structure_info_list, cfg)
            if is_valid: break
        pass  # end while not is_valid:
        block_structure_info_list = new_block_structure_info_list

    return block_structure_info_list


def do_main_job(popu_nas, model_nas, logger=None, max_iter=None, cfg=None,
                masternet_structure_info=None,):

    # whether to fix the stage layer, enable minor_mutation for mutation function.
    if cfg.space_minor_mutation and popu_nas.num_evaluated_nets_count > cfg.space_minor_iter:
        minor_mutation = True
    else:
        minor_mutation = False

    for loop_count in range(max_iter):
        # too many networks in the population pool, remove one with the smallest accuracy
        if len(popu_nas.popu_structure_list) > cfg.ea_popu_size:
            logger.debug('*** debug: rank={}, population too large, remove some.'.format(cfg.rank))
            popu_nas.rank_population(maintain_popu=True)
        pass

        # ----- begin random generate a new structure and examine its performance ----- #
        logger.debug('*** debug: rank={}, generate random structure, loop_count={}'.format(cfg.rank, loop_count))
        if len(popu_nas.popu_structure_list) == 0:
            random_structure_info = masternet_structure_info
        else:
            init_random_structure_info = random.choice(popu_nas.popu_structure_list)
            random_structure_info = get_new_random_structure_info(
                block_structure_info_list=init_random_structure_info,
                mutate_function=model_nas.mutation, cfg=cfg, minor_mutation=minor_mutation)
        pass  # end if
        logger.debug('*** debug: rank={}, random structure generated'.format(cfg.rank))

        # load random_structure_info, get the basic info, update the population
        random_struct_info = model_nas.get_info_for_evolution(structure_info=random_structure_info)
        if random_struct_info["is_satify_budget"]: popu_nas.update_population(random_struct_info)

    pass  # end for loop_count

    logger.debug('*** debug: rank={}, cleaning population before return main_job'.format(cfg.rank))
    popu_nas.rank_population(maintain_popu=True)
    logger.debug('*** debug: rank={}, return main_job'.format(cfg.rank))

    return popu_nas


def main():
    args = parse_args()
    Config = load_py_module_from_path(args.config+":Config")
    cfg = Config()

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.cfg_options is not None:
        cfg.merge(args.cfg_options)
        cfg.config_check()

    if cfg.ea_dist_mode == 'single':
        cfg.gpu = 0
        cfg.world_size = 1
        cfg.rank = 0

    elif cfg.ea_dist_mode == 'mpi':
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        cfg.world_size = mpi_size
        cfg.rank = mpi_rank
        if cfg.score_type=="madnas" and not cfg.lat_gpu:
            cfg.gpu = None
        else:
            auto_gpu = AutoGPU()
            cfg.gpu = auto_gpu.gpu
        random.seed(13 + mpi_rank)
    else:
        raise RuntimeError('Not implemented dist_mode=' + cfg.ea_dist_mode)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, "search_log/log_rank%d_%s"%(cfg.rank, timestamp))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = get_root_logger(name='Search', rank=cfg.rank, log_file=log_file, log_level=cfg.log_level)
    logger.info('Environment info:\n%s\n'%(str(cfg)))

    # finished searching, then exist
    best_structure_txt = os.path.join(cfg.work_dir, 'best_structure.txt')
    if os.path.isfile(best_structure_txt) and not cfg.only_master:
        print('skip ' + best_structure_txt)
        return
    # copy config
    if os.path.isfile(args.config) and cfg.rank==0:
        # os.system("cp %s %s/"%(args.config, cfg.work_dir))
        save_pyobj(os.path.join(cfg.work_dir, 'config_nas.txt'), cfg)

    # begin to build the masternet
    logger.info('begin to build the masternet and population:\n')
    model_nas = BuildNAS(cfg, logger)
    popu_nas = Population(cfg, logger)

    # TODO: cfg.entropy_flow_prior = [float(x) for x in cfg.entropy_flow_prior.split(',')]

    # load masternet and get the basic info
    masternet_info = model_nas.get_info_for_evolution(structure_txt=cfg.space_structure_txt, flop_thop=True)
    masternet_structure_info = masternet_info["structure_info"]
    logger.info(masternet_info)
    if not masternet_info["is_satify_budget"]:
        raise ValueError("The initial network must meet the limit budget, preferably less than 1/4")
    if cfg.only_master: exit()

    # initialize the population with the masternet
    for i in range(popu_nas.popu_size):
        popu_nas.update_population(masternet_info)

    sync_interval = round(cfg.ea_sync_size_ratio * cfg.ea_popu_size)
    logger.info('\nsync_interval={}'.format(sync_interval))

    num_evaluated_nets_count = 0
    # load population list
    if cfg.ea_load_population is not None:
        logger.info('load_population= %s'%(cfg.ea_load_population))
        loader = load_pyobj(cfg.ea_load_population)
        popu_nas.merge_shared_data(loader)

    start_timer = time.time()
    worker_busy_list = [False] * cfg.world_size
    worker_req_list = [None] * cfg.world_size
    last_export_generation_iteration = 0

    early_stop = False
    last_min_score = -1
    last_min_score_step = 0
    while not early_stop:
        # early stop when min score stops update largely
        if cfg.rank == 0 and popu_nas.num_evaluated_nets_count >= 10000:
            min_score = min(popu_nas.popu_acc_list)
            max_score = max(popu_nas.popu_acc_list)
            if min_score - last_min_score < max_score * 1e-3:
                if popu_nas.num_evaluated_nets_count - last_min_score_step > 0.2 * cfg.ea_num_random_nets:
                    early_stop = False # no early stop
                    # early_stop = True # early stop is remained for madnas
                    # logger.info('early stop since min_score={:.4g} from iter={} to iter={}'.format(min_score, last_min_score_step, popu_nas.num_evaluated_nets_count))
            else:
                last_min_score = min_score
                last_min_score_step = popu_nas.num_evaluated_nets_count


        # for master node, gather all worker results, if any
        if cfg.rank == 0:
            for worker_id in range(1, cfg.world_size):
                if worker_busy_list[worker_id]:
                    the_req = worker_req_list[worker_id]
                    req_status, req_item = the_req.test()
                    if req_status:
                        global_shared_data = req_item
                        the_req.wait()
                        logger.debug('*** master recv results from work {}, len={}, n={}'.format(worker_id,
                                                                                             len(popu_nas.popu_structure_list),
                                                                                             popu_nas.num_evaluated_nets_count))
                        if global_shared_data is not None:  # when worker send non-empty list
                            popu_nas.merge_shared_data(global_shared_data, update_num=False)
                        else:
                            raise RuntimeError('from worker {}, recv None results!'.format(worker_id))

                        logger.debug('*** master updates n from {} to {}'.format(popu_nas.num_evaluated_nets_count,
                                                                                       popu_nas.num_evaluated_nets_count + sync_interval))
                        popu_nas.num_evaluated_nets_count += sync_interval # updat the num_evaluted after finish once sync
                        worker_req_list[worker_id] = None
                        worker_busy_list[worker_id] = False
                    pass
                pass
            pass  # end for worker_id
        pass  # end cfg.rank == 0:

        # for worker node, ask for new jobs
        if cfg.rank > 0:
            buf = bytearray(1 << 28)
            req = mpi_comm.irecv(buf, source=0, tag=1)
            global_shared_data = req.wait()
            # print("global_shared_data", global_shared_data)
            logger.debug('*** debug: worker {} is assigned new jobs, len={}, n={}.'.format(cfg.rank,
                                                                               len(popu_nas.popu_structure_list),
                                                                                popu_nas.num_evaluated_nets_count))
            if global_shared_data is not None: popu_nas.merge_shared_data(global_shared_data)

        # enough jobs done, master node clean up and exit
        if cfg.rank == 0 and (popu_nas.num_evaluated_nets_count >= cfg.ea_num_random_nets or early_stop):
            logger.debug('*** debug: master send termination signal to all  workers.')
            for worker_id in range(1, cfg.world_size):
                if worker_busy_list[worker_id]:
                    # logger.info('master waiting worker {} to finish last job.'.format(worker_id))
                    the_req = worker_req_list[worker_id]
                    _ = the_req.wait()
                    worker_req_list[worker_id] = None
                    worker_busy_list[worker_id] = False
                    logger.debug('*** debug: master knows that worker {} has finished last job.'.format(worker_id))

                # send done signal to worker and wait for confirmation
                req = mpi_comm.isend(popu_nas.export_dict(), dest=worker_id, tag=1)
                req.wait()
                logger.debug('*** debug: master has send termination signal to worker {}.'.format(worker_id))
            pass  # end for worker_id
            logger.debug('*** debug: master has send termination signal to everyone, master break looping now.')
            break
        pass  # end if

        # enough jobs done, worker node clean up and exit
        if cfg.rank > 0 and (popu_nas.num_evaluated_nets_count >= cfg.ea_num_random_nets or early_stop):
            logger.debug('*** debug: worker {} recv termination signal. Break now.'.format(cfg.rank))
            break

        # for master, assign new jobs to workers
        if cfg.rank == 0:
            for worker_id in range(1, cfg.world_size):
                if not worker_busy_list[worker_id]:
                    req = mpi_comm.isend(popu_nas.export_dict(), dest=worker_id, tag=1)
                    req.wait()
                    logger.debug('*** debug: master assign new job to worker {}. n={}'.format(
                                worker_id, popu_nas.num_evaluated_nets_count))
                    buf = bytearray(1 << 28)
                    req = mpi_comm.irecv(buf, source=worker_id, tag=2)
                    worker_busy_list[worker_id] = True
                    worker_req_list[worker_id] = req
                pass
            pass  # end for worker_id
        pass  # end for

        # rank 0 processes the merge task, so the iteration is smaller than others
        if cfg.rank == 0:
            this_worker_max_iter = max(10, sync_interval // 10)
        else:
            this_worker_max_iter = sync_interval

        logger.debug('*** debug: rank={}, do_main_job() begin.'.format(cfg.rank))
        popu_nas = do_main_job(popu_nas, model_nas, logger=logger, 
            max_iter=this_worker_max_iter, cfg=cfg,
            masternet_structure_info=masternet_structure_info)

        if cfg.rank == 0:
            popu_nas.num_evaluated_nets_count += this_worker_max_iter

        logger.debug('*** debug: rank={}, do_main_job() end.'.format(cfg.rank))

        # for worker node, push result to master
        if cfg.rank > 0:
            req = mpi_comm.isend(popu_nas.export_dict(), dest=0, tag=2)
            req.wait()
            logger.debug('*** debug: worker {} push results to master. n={}.'.format(cfg.rank, popu_nas.num_evaluated_nets_count))

        # export generation
        if cfg.rank == 0 and popu_nas.num_evaluated_nets_count - last_export_generation_iteration > \
                max(1, cfg.ea_log_freq):
            export_generation_filename = os.path.join(cfg.work_dir,
                                                      'nas_cache/iter{}.txt'.format(popu_nas.num_evaluated_nets_count))
            print('exporting generation: %s'%(export_generation_filename))
            save_pyobj(export_generation_filename, popu_nas.export_dict())

            # logging intermediate results
            elasp_time = time.time() - start_timer
            remain_time = elasp_time * float(cfg.ea_num_random_nets - popu_nas.num_evaluated_nets_count) / (
                        1e-10 + float(popu_nas.num_evaluated_nets_count))
            if len(popu_nas.popu_acc_list) > 0:
                individual_info = popu_nas.get_individual_info(idx=0)
                logger.info('---rank={}, n={}, elasp_time={:4g}h, remain_time={:4}h'.format(
                        cfg.rank, popu_nas.num_evaluated_nets_count, elasp_time / 3600, remain_time / 3600))
                logger.info('---best_individual: {}'.format(individual_info))

            last_export_generation_iteration = popu_nas.num_evaluated_nets_count
        pass  # end export generation
    pass  # end while True


    # export results for master node
    if cfg.rank == 0:
        # export final generation
        export_generation_filename = os.path.join(cfg.work_dir, 'nas_cache/iter_final.txt')
        print('exporting generation: ' + export_generation_filename)
        save_pyobj(export_generation_filename, popu_nas.export_dict())

        # export best structure info
        if len(popu_nas.popu_acc_list) > 0:
            individual_info = popu_nas.get_individual_info(idx=0, is_struct=True)
            save_pyobj(best_structure_txt, individual_info["structure"])
            nas_info_txt = os.path.join(cfg.work_dir, 'nas_info.txt')
            save_pyobj(nas_info_txt, individual_info)
        pass  # end with
    pass  # end if cfg.rank == 0
    exit()


if __name__ == '__main__':
    main()