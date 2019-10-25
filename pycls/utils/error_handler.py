#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multiprocessing error handler."""

import os
import signal
import threading


class ChildException(Exception):
    """Wraps an exception from a child process."""

    def __init__(self, child_trace):
        super(ChildException, self).__init__(child_trace)


class ErrorHandler(object):
    """Multiprocessing error handler (based on fairseq's).

    Listens for errors in child processes and
    propagates the tracebacks to the parent process.
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
