from enum import Enum
import json
import logging
import os
import sys
import tarfile
from typing import Optional

from torch.cuda import amp

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module

from common.progress_tracker import ProgressTracker
from common.utils import gpu_id

LOG = logging.getLogger()

class Optimization(Enum):
    nothing = 0
    mxprO0 = 1
    mxprO1 = 2
    mxprO2 = 3
    mxprO3 = 4


AmpOptimizations = {Optimization.mxprO0: "O0",
                    Optimization.mxprO1: "O1",
                    Optimization.mxprO2: "O2",
                    Optimization.mxprO3: "O3"}


def can_make_step(parameters):
    for p in parameters:
        if (p.grad is not None) and (not torch.all(torch.isfinite(p.grad.data))):
            return False
    return True


class TorchOptimizerWrapper:

    def __init__(self, params, optimizer: torch.optim.Optimizer, amp_level: AmpOptimizations,
                 max_grad_norm: Optional[float] = None):
        self._optimizer = optimizer
        self._amp_level = amp_level
        self._max_grad_norm = max_grad_norm
        self._params = params
        self._optimizer.zero_grad()
        self._scaler = amp.GradScaler()

    def __call__(self, loss: Tensor):
        if self._amp_level in AmpOptimizations.keys():
            #with amp.scale_loss(loss, self._optimizer) as scaled_loss:
            #    scaled_loss.backward()
            #LOG.debug("before scaling")
            self._scaler.scale(loss).backward()
            #if self._max_grad_norm is not None:
            #    torch.nn.utils.clip_grad_norm_(amp.master_params(self._optimizer), self._max_grad_norm)
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._params, self._max_grad_norm)
            #LOG.debug("before step")
            self._scaler.step(self._optimizer)
            #LOG.debug("before update")
            self._scaler.update()
        else:
            loss.backward()
            if self._max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._optimizer.param_groups[0]["params"], self._max_grad_norm)
            if can_make_step(master_params(self._optimizer)):
                self._optimizer.step()
        self._optimizer.zero_grad()

    @property
    def optimizer(self):
        return self._optimizer

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        self._optimizer.load_state_dict(state)
    
    def change_lr(self, new_lr):
        for g in self._optimizer.param_groups:
            g['lr'] = new_lr


def set_learning_rate(optimizer: torch.optim, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def nccl_barrier_on_cpu():
    if not torch.distributed.is_initialized():
        return
    with torch.no_grad():
        fake = torch.zeros([16])
        fake[0] = 10
        fake = fake.cuda()
        torch.distributed.all_reduce(fake)
        fake = fake.cpu()


def all_reduce(value: float):
    if dist.is_initialized():
        value_tensor = torch.tensor(value).cuda()
        dist.all_reduce(value_tensor)
        value = value_tensor.cpu().item()
        return value
    else:
        return value


# TODO(noxoomo): hard code tmp is not good
def reduce_reader_progresses(reader_progresses, iteration):
    # if not torch.distributed.is_initialized():
    return reader_progresses

    my_rank = torch.distributed.get_rank()
    # dirty way to sync readers and save them in one snapshot
    with open("/tmp/reader_progress_snapshot_rank_{}_{}.json".format(my_rank, iteration), "w") as f:
        states = [reader_progress.save() for reader_progress in reader_progresses]
        f.write(json.dumps(states))

    nccl_barrier_on_cpu()

    total_ranks = torch.distributed.get_world_size()
    if my_rank == 0:
        for rank in range(0, total_ranks):
            with open(f"/tmp/reader_progress_snapshot_rank_{rank}_{iteration}.json", "r") as f:
                states = json.loads(f.read())
                assert len(states) == len(reader_progresses)
                for i, state in enumerate(states):
                    # TODO: here we use implementation detail for readers, should be fixed
                    # reader_progress.filter_ranks(rank, torch.distributed.get_world_size())
                    # keys_to_drop = [x for x in state.keys() if x % total_ranks != rank]
                    # for key in keys_to_drop:
                    #     states.pop(key)
                    reader_progresses[i].merge_progress(state, filter_old=True)
    nccl_barrier_on_cpu()
    return reader_progresses


def save_snapshot(model: torch.nn.Module, optimizer: TorchOptimizerWrapper, scheduler,
                  progress_tracker: ProgressTracker,
                  snapshot_path: str, log_dir: str, log_archive_path: str):
    LOG.info('Saving snapshot')
    with torch.no_grad():
        sys.stderr.write(f"saving snapshot rank {gpu_id()}...\n")
        nccl_barrier_on_cpu()
        sys.stderr.write(f"after barrier\n")
        reduced_progress = progress_tracker.save_distributed()
        sys.stderr.write(f"after save distributed\n")
        reduced_progress["log_dir"] = log_dir
        if gpu_id() == 0:
            try:
                model_to_save = model.module if hasattr(model, "module") else model  # Only save the model it-self
                save_checkpoint = {
                    "progress_tracker_state": reduced_progress,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None
                }

                snapshot_tmp = snapshot_path + ".bac"
                torch.save(save_checkpoint, snapshot_tmp)
                os.rename(snapshot_tmp, snapshot_path)
                
                sys.stderr.write(f"after save model\n")

                sys.stderr.write("Snapshot was saved...\n")
            except Exception as e:
                sys.stderr.write(f"Can't save snapshot: {e}\n")
                raise e

        nccl_barrier_on_cpu()


        
def get_latest_snapshot_path(snapshots_dir: str):
    path = None
    for subdir, dirs, files in os.walk(snapshots_dir):
        if subdir != snapshots_dir:
            continue
        for file in files:
            if not file.startswith('snapshot'):
                continue
            if path is None or int(file.split("snapshot")[-1]) > int(path.split("snapshot")[-1]):
                path = os.path.join(subdir, file)
    return path



def restore_from_snapshot(snapshots_dir: str, model: Module,
                          optimizer: Optional[TorchOptimizerWrapper] = None, scheduler: Optional = None,
                          progress: Optional[ProgressTracker] = None, saver: Optional = None, snapshot_path=None):
    if snapshot_path is None:
        snapshot_path = get_latest_snapshot_path(snapshots_dir)
    LOG.info('Restoring snapshot from {}'.format(snapshot_path))
    
    nccl_barrier_on_cpu()
    checkpoint = torch.load(snapshot_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint["scheduler_state"]:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if progress:
        progress.load(checkpoint["progress_tracker_state"])

    LOG.info("Restored from snapshot")
    nccl_barrier_on_cpu()

    if progress and saver:
        saver._iteration = progress._total_iterations


class ProgressListener:
    def on_train_freq(self, progress_tracker: ProgressTracker, batch, loss, output):
        pass

    def on_test_freq(self, progress_tracker: ProgressTracker):
        pass

    def after_finish_batch(self):
        pass

    def after_finish_epoch(self):
        pass

    def after_finish_batch_block(self):
        pass


class ProgressListeners(ProgressListener):
    def __init__(self):
        self._all = []

    def add(self, listener: ProgressListener):
        self._all.append(listener)

    def on_train_freq(self, progress_tracker: ProgressTracker, batch, loss, output):
        for listener in self._all:
            listener.on_train_freq(progress_tracker, batch, loss, output)

    def on_test_freq(self, progress_tracker: ProgressTracker):
        for listener in self._all:
            listener.on_test_freq(progress_tracker)

    def after_finish_batch(self):
        for listener in self._all:
            listener.after_finish_batch()

    def after_finish_epoch(self):
        for listener in self._all:
            listener.after_finish_epoch()
            
    def after_finish_batch_block(self):
        for listener in self._all:
            listener.after_finish_batch_block()


def create_save_path(save_dir, iter_num, max_shapshots_num=30):
    if gpu_id() != 0:
        return None
    save_path = os.path.join(save_dir, 'snapshot' + str(iter_num))
    num_of_snapshots = sum([1 for _, _, files in os.walk(save_dir) for f in files if f.startswith('snapshot')])
    if num_of_snapshots >= max_shapshots_num:
        raise Exception('Snapshots found: {}, close to disk memory'.format(num_of_snapshots))
    else:
        LOG.info('Snapshots found: {}, saving new snapshot...'.format(num_of_snapshots))
    return save_path


class ProgressSaver(ProgressListener):
    def __init__(self,
                 progress_tracker: ProgressTracker,
                 model,
                 optimizer,
                 scheduler,
                 safe_freq_iterations,
                 safe_path,
                 log_dir,
                 log_archive_path):
        self._progress_tracker = progress_tracker
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._safe_freq_iterations = safe_freq_iterations
        self._iteration = 0
        self._snapshot_path = safe_path
        self._log_dir = log_dir
        self._logs_archive_path = log_archive_path
        self._time_to_save = False

    def after_finish_batch(self):
        if (self._iteration + 1) % self._safe_freq_iterations == 0:
            self._time_to_save = True
        self._iteration += 1

    def after_finish_batch_block(self):
        if self._time_to_save:
            LOG.info('ProgressSaver: time_to_save')
            self.__save()
            self._time_to_save = False

    def after_finish_epoch(self):
        self.__save()

    def on_training_finish(self):
        self.__save()

    def __save(self):
        save_path = create_save_path(self._snapshot_path, self._iteration)
        save_snapshot(self._model, self._optimizer, self._scheduler, self._progress_tracker, save_path,
                      self._log_dir, self._logs_archive_path)
