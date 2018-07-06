from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from torch import optim
from utils.config import cfg

# ---------------------------------------------------------------------------- #
# 
# ---------------------------------------------------------------------------- #

def config_optimizer(param):
    """
    config optimizer
    """
    logger = logging.getLogger(__name__)
    logger.info('using {}: base_learning_rate = {}, momentum = {}, weight_decay = {}'. 
                format(cfg.SOLVER.OPTIMIZER, cfg.SOLVER.BASE_LR, cfg.SOLVER.MOMENTUM, cfg.SOLVER.WEIGHT_DECAY))
    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(param, lr=cfg.SOLVER.BASE_LR, 
                    momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(param, lr=cfg.SOLVER.BASE_LR,
                    momentum=cfg.SOLVER.MOMENTUM, alpha=cfg.SOLVER.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(param, lr=cfg.SOLVER.BASE_LR,
                    betas=(cfg.SOLVER.MOMENTUM, cfg.SOLVER.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        AssertionError('optimizer can not be recognized.')
    return optimizer

