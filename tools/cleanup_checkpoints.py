#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

import pycls.core.checkpoint as cp


def parse_args():
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="Cleanup checkpoints.")
    help_s = "Root dir for cleanup"
    parser.add_argument("--dir", help=help_s, required=True, type=str)
    help_s, choices, dft = "Cleanup mode", ["last", "none"], "last"
    parser.add_argument("--keep", help=help_s, choices=choices, default=dft, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    for rootdir, subdirs, _ in os.walk(args.dir):
        for subdir in subdirs:
            dir = os.path.join(rootdir, subdir)
            files = [x for x in os.listdir(dir) if x.endswith(".pyth")]
            if "/checkpoints" not in dir or not len(files):
                continue
            print("\nWorking dir:", dir)
            msg = ["{:03d}: {}".format(idx, file) for idx, file in enumerate(files)]
            print("\n".join(["Checkpoints:"] + msg))
            print("\nCleanup {}\n[yes/n]? (Keep: {} checkpoints)".format(dir, args.keep))
            if input().lower() == "yes":
                cp.delete_checkpoints(dir, keep=args.keep)


if __name__ == "__main__":
    main()
