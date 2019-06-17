#!/usr/bin/env python3

"""Logging."""

# TODO(ilijar): support logging json stats to file

import builtins
import decimal
import logging
import simplejson
import sys

from pycls.core.config import cfg

import pycls.utils.distributed as du

# Show filename and line number in logs
_FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'


def _suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=' ', _end='\n', _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def setup_logging():
    """Sets up the logging."""
    # Enable logging only for the master process
    if du.is_master_proc():
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []
        # Construct logging configuration
        logging_config = {
            'level': logging.INFO,
            'format': _FORMAT
        }
        # Log either to stdout or to a file
        if cfg.LOG_DEST == 'stdout':
            logging_config['stream'] = sys.stdout
        else:
            logging_config['filename'] = cfg.LOG_DEST
        # Configure logging
        logging.basicConfig(**logging_config)
    else:
        _suppress_print()


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def log_json_stats(stats):
    """Logs json stats."""
    # It seems that in Python >= 3.5 json.encoder.FLOAT_REPR has no effect
    # Use decimal+string as a workaround for having fixed length values in logs
    stats = {
        k: decimal.Decimal('{:.6f}'.format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    print('json_stats: {:s}'.format(json_stats))
