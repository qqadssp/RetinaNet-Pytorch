from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
from utils.config import cfg
from datasets.VOC import VOCdataset

datasets_map = {
                    'VOC2012': VOCdataset              
                }


def load_data(datasets=('VOC2012', 'train.txt')):
    
    logger = logging.getLogger(__name__)
    logger.info('load_train_data: {}, {}'. format(datasets[0], datasets[1]))

    trainset = datasets_map[datasets[0]]()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    return trainloader

# test()
