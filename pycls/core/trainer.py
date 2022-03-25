#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import contextlib
import os
import random
from copy import deepcopy

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as data_loader
import torch
import torch.cuda.amp as amp
from fairscale.nn import auto_wrap, config_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.grad_scaler import ShardedGradScaler
from pycls.core.config import cfg
from pycls.core.io import cache_url, pathmgr


logger = logging.get_logger(__name__)


def auto_wrap_ln(module, fsdp_config=None, wrap_it=True, assert_on_collision=True):
    """
    Auto wrap all LayerNorm instances with a safer FSDP, esp. when convert
    to LayerNorm is used and the outer FSDP is flattening.
    We put KN in is own full precision, unflatten, single GPU group FSDP.  Note, LNs still have
    a group size == world_size. The input and output for LN are still FP16 in mixed precision mode.
    See ``keep_batchnorm_fp32`` here: https://nvidia.github.io/apex/amp.html
    This needs to be done at each rank, like models being wrapped by FSDP at each rank.
    Args:
        module (nn.Module):
            The model (or part of the model) in which BN to be pre-wrapped.
        process_group (ProcessGroup):
            Optional process group to be used.
        fsdp_config (Dict):
            Optional fsdp_config to be used.
        wrap_it (bool):
            Whether or not wrap the module after setting the config.
            Default: True
        assert_on_collision (bool):
            Whether or not assert if a wrapper_config already exists on the module.
            Default: True
    Returns:
        Processed module, where BNs are wrapped with a special FSDP instance.
    """
    if fsdp_config is None:
        fsdp_config = {
            "mixed_precision": False,  # Keep the weights in FP32.
            "flatten_parameters": False,  # Do not flatten.
            # Reshard==False is good for performance. When FSDP(checkpoint(FSDP(bn))) is used, this
            # **must** be False because BN's FSDP wrapper's pre-backward callback isn't called
            # within the checkpoint's outer backward when multiple forward passes are used.
            "reshard_after_forward": False,
            # No bucketing or small bucketing should be enough for BNs.
            "bucket_cap_mb": 0,
            # Setting this for SyncBatchNorm. This may have a performance impact. If
            # SyncBatchNorm is used, this can be enabled by passing in the `fsdp_config` argument.
            "force_input_to_fp32": False,
        }

    # Assign the config dict to BNs.
    for m in module.modules():
        if isinstance(m, torch.nn.LayerNorm):
            if assert_on_collision:
                assert not hasattr(
                    m, "wrapper_config"
                ), "Module shouldn't already have a wrapper_config. Is it tagged already by another policy?"
            m.wrapper_config = fsdp_config

    # Wrap it.
    with (
        enable_wrap(config_auto_wrap_policy, wrapper_cls=FSDP)
        if wrap_it
        else contextlib.suppress()
    ):
        return auto_wrap(module)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_main_proc():
        # Ensure that the output dir exists
        pathmgr.mkdirs(cfg.OUT_DIR)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log torch, cuda, and cudnn versions
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    env = "".join([f"{key}: {value}\n" for key, value in sorted(os.environ.items())])
    logger.info(f"os.environ:\n{env}")
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()

    if cfg.FSDP.ENABLED:
        fsdp_config = {}
        fsdp_config["reshard_after_forward"] = cfg.FSDP.RESHARD_AFTER_FW
        fsdp_config["mixed_precision"] = cfg.TRAIN.MIXED_PRECISION
        if cfg.TRAIN.MIXED_PRECISION:
            fsdp_config["clear_autocast_cache"] = True

        # Enable LAYER_NORM_FP32 wrapping for mixed precision training only. It is not
        # required for full precision training.
        ema_model = builders.build_model()
        if cfg.TRAIN.MIXED_PRECISION and cfg.FSDP.LAYER_NORM_FP32:

            def do_wrap(m):
                m = auto_wrap_ln(m)
                return FSDP(m, **fsdp_config)

            model = do_wrap(model)
            ema_model = do_wrap(ema_model)
        else:
            with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
                model = FSDP(model, **fsdp_config)
                ema_model = FSDP(ema_model, **fsdp_config)

    else:
        # Build the model
        model = builders.build_model()
        ema_model = deepcopy(model)

    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    ema_model = ema_model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1 and not cfg.FSDP.ENABLED:
        # Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(module=model, device_ids=[cur_device], output_device=cur_device)
    return model, ema_model


def get_weights_file(weights_file):
    """Download weights file if stored as a URL."""
    download = dist.is_main_proc(local=True)
    weights_file = cache_url(weights_file, cfg.DOWNLOAD_CACHE, download=download)
    if cfg.NUM_GPUS > 1:
        torch.distributed.barrier()
    return weights_file


def train_epoch(loader, model, ema, loss_fun, optimizer, scaler, meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    ema.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Convert labels to smoothed one-hot vector
        labels_one_hot = net.smooth_one_hot_labels(labels)
        # Apply mixup to the batch (no effect if mixup alpha is 0)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        # Perform the forward pass and compute the loss
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            loss = loss_fun(preds, labels_one_hot)
        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Update ema weights
        net.update_model_ema(model, ema, cur_epoch, cur_iter)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


@torch.no_grad()
def test_epoch(loader, model, meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, ema, loss_fun, and optimizer
    model, ema = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cp.has_checkpoint():
        file = cp.get_last_checkpoint()
        epoch = cp.load_checkpoint(file, model, ema, optimizer)[0]
        logger.info("Loaded checkpoint from: {}".format(file))
        start_epoch = epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        train_weights = get_weights_file(cfg.TRAIN.WEIGHTS)
        cp.load_checkpoint(train_weights, model, ema)
        logger.info("Loaded initial weights from: {}".format(train_weights))
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    ema_meter = meters.TestMeter(len(test_loader), "test_ema")
    # Create a GradScaler for mixed precision training
    if cfg.FSDP.ENABLED:
        scaler = ShardedGradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    else:
        scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, ema, loss_fun, optimizer, scaler, train_meter)
        train_epoch(*params, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
            net.compute_precise_bn_stats(ema, train_loader)
        # Evaluate the model
        test_epoch(test_loader, model, test_meter, cur_epoch)
        test_epoch(test_loader, ema, ema_meter, cur_epoch)
        test_err = test_meter.get_epoch_stats(cur_epoch)["top1_err"]
        ema_err = ema_meter.get_epoch_stats(cur_epoch)["top1_err"]
        # Save a checkpoint
        file = cp.save_checkpoint(model, ema, optimizer, cur_epoch, test_err, ema_err)
        logger.info("Wrote checkpoint to: {}".format(file))


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model, _ = setup_model()
    # Load model weights
    test_weights = get_weights_file(cfg.TEST.WEIGHTS)
    cp.load_checkpoint(test_weights, model)
    logger.info("Loaded model weights from: {}".format(test_weights))
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model, _ = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Compute model and loader timings
    benchmark.compute_time_model(model, loss_fun)


def time_model_and_loader():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model, _ = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
