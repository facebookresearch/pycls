#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed helpers."""

import multiprocessing
import os
import signal
import threading
import traceback

import torch
from pycls.core.config import cfg


def is_master_proc():
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints. In
    the multi GPU setting, we assign the master role to the rank 0 process. When
    training using a single GPU, there is a single process which is considered master.
    """
    return cfg.NUM_GPUS == 1 or torch.distributed.get_rank() == 0


def init_process_group(proc_rank, world_size):
    """Initializes the default process group."""
    # Set the GPU to use
    torch.cuda.set_device(proc_rank)
    # Initialize the process group
    torch.distributed.init_process_group(
        backend=cfg.DIST_BACKEND,
        init_method="tcp://{}:{}".format(cfg.HOST, cfg.PORT),
        world_size=world_size,
        rank=proc_rank,
    )


def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()


def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group (equivalent to cfg.NUM_GPUS).
    """
    # There is no need for reduction in the single-proc case
    if cfg.NUM_GPUS == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS)
    return tensors


class ChildException(Exception):
    """Wraps an exception from a child process."""

    def __init__(self, child_trace):
        super(ChildException, self).__init__(child_trace)


class ErrorHandler(object):
    """Multiprocessing error handler (based on fairseq's).

    Listens for errors in child processes and propagates the tracebacks to the parent.
    """

    def __init__(self, error_queue):
        # Shared error queue
        self.error_queue = error_queue
        # Children processes sharing the error queue
        self.children_pids = []
        # Start a thread listening to errors
        self.error_listener = threading.Thread(target=self.listen, daemon=True)
        self.error_listener.start()
        # Register the signal handler
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """Registers a child process."""
        self.children_pids.append(pid)

    def listen(self):
        """Listens for errors in the error queue."""
        # Wait until there is an error in the queue
        child_trace = self.error_queue.get()
        # Put the error back for the signal handler
        self.error_queue.put(child_trace)
        # Invoke the signal handler
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, _sig_num, _stack_frame):
        """Signal handler."""
        # Kill children processes
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)
        # Propagate the error from the child process
        raise ChildException(self.error_queue.get())


def run(proc_rank, world_size, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        init_process_group(proc_rank, world_size)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except KeyboardInterrupt:
        # Killed by the parent process
        pass
    except Exception:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        # Destroy the process group
        destroy_process_group()


def multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs=None):
    """Runs a function in a multi-proc setting (unless num_proc == 1)."""
    # There is no need for multi-proc in the single-proc case
    fun_kwargs = fun_kwargs if fun_kwargs else {}
    if num_proc == 1:
        fun(*fun_args, **fun_kwargs)
        return
    # Handle errors from training subprocesses
    error_queue = multiprocessing.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    # Run each training subprocess
    ps = []
    for i in range(num_proc):
        p_i = multiprocessing.Process(
            target=run, args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)
    # Wait for each subprocess to finish
    for p in ps:
        p.join()
