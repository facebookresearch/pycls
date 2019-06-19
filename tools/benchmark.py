#!/usr/bin/env python3

"""Benchmark different training components."""

# TODO(ilijar): port benchmarking functions from nms

import argparse
import numpy as np
import os
import sys

import torch

from pycls.core.config import assert_cfg
from pycls.core.config import cfg
from pycls.core.config import dump_cfg
from pycls.datasets import loader
from pycls.models import model_builder
from pycls.utils.timer import Timer

import pycls.models.losses as losses
import pycls.models.optimizer as optim
import pycls.utils.logging as lu
import pycls.utils.multiprocessing as mpu

logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark a classification model'
    )
    parser.add_argument(
        '--num-iter',
        dest='num_iter',
        help='Number of iterations to perform',
        default=5000,
        type=int
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def time_train(num_iter, data_loader, model, loss_fun, optimizer):
    """Measures train iter (forward + backward + update) time."""
    assert num_iter <= len(data_loader), \
        'Timing train for more than one epoch not supported'

    # Forward a dummy minibatch to warm up the caching allocator
    # and to synch the processes in the multi-proc setting
    dummy_inputs, _dummy_labels = next(data_loader.__iter__())
    model(dummy_inputs.cuda())

    logger.info('==> Running {} train iters...'.format(num_iter))
    iter_timer = Timer()
    iter_timer.tic()

    # Construct the iterator manually
    data_iterator = data_loader.__iter__()

    for _cur_iter in range(num_iter):
        # Get the next minibatch
        inputs, labels = next(data_iterator)
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Measure iter time
        torch.cuda.synchronize()
        iter_timer.toc()
        iter_timer.tic()

    logger.info('Average iter time: {:.6f}s'.format(iter_timer.average_time))


def time_forward(num_iter, data_loader, model, loss_fun):
    """Measures forward pass time."""
    assert num_iter <= len(data_loader), \
        'Timing forward for more than one epoch not supported'

    # Forward a dummy minibatch to warm up the caching allocator
    # and to synch the processes in the multi-proc setting
    dummy_inputs, _dummy_labels = next(data_loader.__iter__())
    model(dummy_inputs.cuda())

    logger.info('==> Running {} forward iters...'.format(num_iter))
    iter_timer = Timer()
    iter_timer.tic()

    # Construct the iterator manually
    data_iterator = data_loader.__iter__()

    for _cur_iter in range(num_iter):
        # Get the next minibatch
        inputs, labels = next(data_iterator)
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        loss_fun(preds, labels)
        # Measure iter time
        torch.cuda.synchronize()
        iter_timer.toc()
        iter_timer.tic()

    logger.info('Average iter time: {:.6f}s'.format(iter_timer.average_time))


def time_train_wo_loading(num_iter, data_loader, model, loss_fun, optimizer):
    """Measures train iter (forward + backward + update) w/o loading time."""

    # Use fixed data for each iteration
    inputs, labels = next(data_loader.__iter__())
    # Transfer the data to the current GPU device
    inputs, labels = inputs.cuda(), labels.cuda()

    # Forward the data once to warm up the caching allocator
    # and to synch the processes in the multi-proc setting
    model(inputs)

    logger.info('==> Running {} train w/o loading iters...'.format(num_iter))
    iter_timer = Timer()
    iter_timer.tic()

    for _cur_iter in range(num_iter):
        # Perform the forward pass
        preds = model(inputs)
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Measure iter time
        torch.cuda.synchronize()
        iter_timer.toc()
        iter_timer.tic()

    logger.info('Average iter time: {:.6f}s'.format(iter_timer.average_time))


def time_forward_wo_loading(num_iter, data_loader, model, loss_fun):
    """Measures forward pass w/o loading time."""

    # Use fixed data for each iteration
    inputs, labels = next(data_loader.__iter__())
    # Transfer the data to the current GPU device
    inputs, labels = inputs.cuda(), labels.cuda()

    # Forward the data once to warm up the caching allocator
    # and to synch the processes in the multi-proc setting
    model(inputs)

    logger.info('==> Running {} forward w/o loading iters...'.format(num_iter))
    iter_timer = Timer()
    iter_timer.tic()

    for _cur_iter in range(num_iter):
        # Perform the forward pass
        preds = model(inputs)
        loss_fun(preds, labels)
        # Measure iter time
        torch.cuda.synchronize()
        iter_timer.toc()
        iter_timer.tic()

    logger.info('Average iter time: {:.6f}s'.format(iter_timer.average_time))


def benchmark_model(num_iter):
    """Benchmarks the model."""

    # Construct the data loader
    data_loader = loader.construct_train_loader()
    # Construct the model
    model = model_builder.build_model()
    model.train()
    # Define the loss function
    loss_fun = losses.get_loss_fun()
    # Construct the optimizer
    optimizer = optim.construct_optimizer(model)

    # Benchmarks w/ data loading
    time_train(num_iter, data_loader, model, loss_fun, optimizer)
    time_forward(num_iter, data_loader, model, loss_fun)

    # Benchmarks w/o data loading
    time_train_wo_loading(num_iter, data_loader, model, loss_fun, optimizer)
    time_forward_wo_loading(num_iter, data_loader, model, loss_fun)


def single_proc_benchmark(num_iter):
    """Performs single process benchmarking."""

    # Setup logging
    lu.setup_logging()
    # Show the config
    logger.info('Config:\n{}'.format(cfg))

    # Set RNG seeds for reproducibility (see RNG comment in core/config.py)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Benchmark the model
    benchmark_model(num_iter)


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()

    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the config
    dump_cfg()

    # Run benchmarks
    if cfg.NUM_GPUS > 1:
        mpu.multi_proc_run(
            num_proc=cfg.NUM_GPUS,
            fun=single_proc_benchmark,
            fun_args=(args.num_iter,)
        )
    else:
        single_proc_benchmark(args.num_iter)


if __name__ == '__main__':
    main()
