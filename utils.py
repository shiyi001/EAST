#encoding: utf8
from __future__ import division

import os
import logging
import math

logs = set()
def init_log(name, level = logging.INFO):
    if (name, level) in logs: return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = '%(asctime)s-rk{}-%(filename)s#%(lineno)d:%(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count