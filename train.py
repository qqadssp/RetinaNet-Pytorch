from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import pprint
import sys

from utils.logging import setup_logging
from utils.config import merge_cfg_from_file
from utils.config import cfg
from utils.train_net import train_model

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    train_model()

if __name__ == '__main__':
    main()
