#!/usr/bin/env python3

"""Multiprocessing helpers."""

import multiprocessing as mp
import traceback

from pycls.utils.error_handler import ErrorHandler

import pycls.utils.distributed as du


def run(proc_rank, world_size, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        du.init_process_group(proc_rank, world_size)
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
        du.destroy_process_group()


def multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs={}):
    """Runs a function in a multi-proc setting."""

    # Handle errors from training subprocesses
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Run each training subprocess
    ps = []
    for i in range(num_proc):
        p_i = mp.Process(
            target=run,
            args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)

    # Wait for each subprocess to finish
    for p in ps:
        p.join()
