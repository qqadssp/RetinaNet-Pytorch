from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import sys

import torch
from utils.config import cfg
from modeling.model_builder import create
from modeling.optimizer import config_optimizer
from modeling.loss import FocalLoss
from datasets.datagen import load_data

def train_model():
    """Model training loop."""
    logger = logging.getLogger(__name__)

    model, weights, start_iter = create_train_model()
    logger.info(model)
    setup_train_model(model, weights, train = True)

    trainloader = load_data(cfg.TRAIN.DATASETS)
    optimizer = config_optimizer(param = model.parameters())
    criterion = FocalLoss()

    for cur_iter in range(start_iter, cfg.SOLVER.MAX_ITER):
        logger.info('Epoch: {}'. format(cur_iter))
        total_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            loc_targets = torch.autograd.Variable(loc_targets.cuda())
            cls_targets = torch.autograd.Variable(cls_targets.cuda())

            optimizer.zero_grad()
            loc_preds, cls_preds = model(inputs)
            loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 20 == 0 : 
                logger.info('batch_idx: {}, loc_loss: {}, cls_loss: {}, train_loss: {}, avg_loss: {}'. 
                            format(batch_idx, loc_loss.item(), cls_loss.item(), loss.item(), total_loss/(batch_idx+1)))

        if cur_iter % cfg.TRAIN.SAVE_CHECKPOINTS_EPOCHS == 0:
            logger.info('Saving check point at iter: {}'. format(cur_iter))
            state = {
                'net': model.state_dict(),
                'loss': total_loss/(batch_idx+1),
                'epoch': cur_iter,
            }
            torch.save(state, './checkpoint/' + 'ckpt_' + str(cur_iter) + '.pth')

def create_train_model():
    """Build the model and look for saved model checkpoints in case we can
    resume from one.
    """
    logger = logging.getLogger(__name__)
    start_iter = 0
    weights = torch.load(os.path.join('checkpoint', cfg.TRAIN.WEIGHTS))
    if cfg.TRAIN.AUTO_RESUME: 
        checkpoints = torch.load(os.path.join('checkpoint', cfg.TRAIN.RESUME_FILE))
        start_iter = checkpoints['epoch']
        if start_iter > 0:
            # Override the initialization weights with the found checkpoint
            weights = checkpoints['net']
            logger.info(
                '========> Resuming from checkpoint {} at start iter {}'.
                format(cfg.TRAIN.RESUME_FILE, start_iter)
            )
    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = create(cfg.MODEL.TYPE, cfg.MODEL.CONV_BODY, cfg.MODEL.NUM_CLASSES)
    return model, weights, start_iter

def setup_train_model(model, weights, train = False):
    """Loaded saved weights and create the network in the C2 workspace."""
    logger = logging.getLogger(__name__)
    logger.info('loading weights and setting mode')
    model.load_state_dict(weights)
    if not torch.cuda.is_available() : 
        logger.info('cuda not find')
        sys.exit(1)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    if train : 
        model.train()
        model.module.freeze_bn()
    else : 
        model.val()
