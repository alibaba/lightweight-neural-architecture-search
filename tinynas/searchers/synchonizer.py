# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

import copy
import os
from tinynas.utils.dist_utils import master_only, worker_only, get_mpi_comm

class Synchonizer():
    def __init__(self, world_size, num_random_nets, popu_nas, sync_interval, worker_max_iter,logger):

        self.world_size = world_size
        self.num_random_nets = num_random_nets 
        self.popu_nas = popu_nas
        self.logger = logger
        self.sync_interval = sync_interval 
        self.worker_max_iter = worker_max_iter
        self.mpi_comm = get_mpi_comm()
        self.worker_busy_list = [False] * self.world_size
        self.worker_req_list = [None] * self.world_size

    @master_only 
    def _gather_all_workers_result(self, ): 
        for worker_id in range(1, self.world_size):
            if self.worker_busy_list[worker_id]:
                the_req = self.worker_req_list[worker_id]
                req_status, req_item = the_req.test()
                if req_status:
                    global_shared_data = req_item
                    the_req.wait()
                    self.logger.debug(
                        'master recv results from work {}, len={}, n={}'.
                        format(worker_id,
                            len(self.popu_nas.popu_structure_list),
                            self.popu_nas.num_evaluated_nets_count))
                    if global_shared_data is not None:  # when worker send non-empty list
                        self.popu_nas.merge_shared_data(
                            global_shared_data, update_num=False)
                    else:
                        raise RuntimeError(
                            'from worker {}, recv None results!'.format(
                                worker_id))

                    self.logger.debug('master updates n from {} to {}'.format(
                        self.popu_nas.num_evaluated_nets_count,
                        self.popu_nas.num_evaluated_nets_count + self.sync_interval))
                    self.popu_nas.num_evaluated_nets_count += self.sync_interval
                    # updat the num_evaluted after finish once sync
                    self.worker_req_list[worker_id] = None
                    self.worker_busy_list[worker_id] = False
                pass
            pass
        pass  # end for worker_id

    @worker_only
    def _worker_req_jobs(self,):
        buf = bytearray(1 << 28)
        req = self.mpi_comm.irecv(buf, source=0, tag=1)
        global_shared_data = req.wait()
        # print("global_shared_data", global_shared_data)
        self.logger.debug(
            'assigned new jobs, len={}, n={}.'.format(
                len(self.popu_nas.popu_structure_list),
                self.popu_nas.num_evaluated_nets_count))
        if global_shared_data is not None:
            self.popu_nas.merge_shared_data(global_shared_data)

    @master_only
    def _master_clean_up(self,):
        self.logger.debug('master send termination signal to all  workers.')
        for worker_id in range(1, self.world_size):
            if self.worker_busy_list[worker_id]:
                        # logger.info('master waiting worker {} to finish last job.'.format(worker_id))
                        the_req = self.worker_req_list[worker_id]
                        _ = the_req.wait()
                        self.worker_req_list[worker_id] = None
                        self.worker_busy_list[worker_id] = False
                        self.logger.debug(
                            'master knows that worker {} has finished last job.'.
                            format(worker_id))

            # send done signal to worker and wait for confirmation
            req = self.mpi_comm.isend(
                      self.popu_nas.export_dict(), dest=worker_id, tag=1)
            req.wait()
            self.logger.debug(
                'master has send termination signal to worker {}.'.format(
                     worker_id))
        pass  # end for worker_id

    @master_only
    def _master_assign_jobs(self,):
        for worker_id in range(1, self.world_size):
            if not self.worker_busy_list[worker_id]:
                req = self.mpi_comm.isend(
                    self.popu_nas.export_dict(), dest=worker_id, tag=1)
                req.wait()
                self.logger.debug(
                    'master assign new job to worker {}. n={}'.format(
                        worker_id, self.popu_nas.num_evaluated_nets_count))
                buf = bytearray(1 << 28)
                req = self.mpi_comm.irecv(buf, source=worker_id, tag=2)
                self.worker_busy_list[worker_id] = True
                self.worker_req_list[worker_id] = req
            pass
        pass  # end for worker_id

    @worker_only
    def _worker_commit_result(self,):
        req = self.mpi_comm.isend(self.popu_nas.export_dict(), dest=0, tag=2)
        req.wait()
        self.logger.debug('push results to master. n={}.'.format(
             self.popu_nas.num_evaluated_nets_count))

    @master_only
    def _master_update_evaluated_nets_count(self,):
        self.popu_nas.num_evaluated_nets_count += self.worker_max_iter

    def sync_and_commit_result(self, popu_nas):
        self.popu_nas = popu_nas
        self._worker_commit_result()
        self._master_update_evaluated_nets_count()
        return self.popu_nas

    def sync_and_assign_jobs(self, popu_nas):
        self.popu_nas = popu_nas 
        # for master node, gather all worker results, if any
        self._gather_all_workers_result()
        # for worker node, ask for new jobs
        self._worker_req_jobs()

        # enough jobs done, master node clean up and exit
        enough_flag = self.popu_nas.num_evaluated_nets_count >= self.num_random_nets
        if enough_flag:
            self._master_clean_up()
            return enough_flag, self.popu_nas  
        pass  # end if

        # for master, assign new jobs to workers
        self._master_assign_jobs()
        return enough_flag, self.popu_nas  
