from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

# model part
from modeling.RetinaNet import RetinaNet
model_map = {
                'RetinaNet': RetinaNet,
            }

# conv_body part
from modeling.ResNet_FPN import FPN50
from modeling.ResNet_FPN import FPN101
networks_map = {
                    'ResNet50_FPN': FPN50,
                    'ResNet101_FPN': FPN101
               }

# ---------------------------------------------------------------------------- #
# Helper functions for building various re-usable network bits
# ---------------------------------------------------------------------------- #

def create(model_type, conv_body, num_classes, train=False):
    """Generic model creation function that dispatches to specific model
    building functions.

    By default, this function will generate a data parallel model configured to
    run on cfg.NUM_GPUS devices. However, you can restrict it to build a model
    targeted to a specific GPU by specifying gpu_id. This is used by
    optimizer.build_data_parallel_model() during test time.
    """
    logger = logging.getLogger(__name__)
    model = model_map[model_type](networks_map[conv_body], num_classes)
    return model

# ---------------------------------------------------------------------------- #
# Network inputs
# ---------------------------------------------------------------------------- #

def add_training_inputs(model, roidb=None):
    """Create network input ops and blobs used for training. To be called
    *after* model_builder.create().
    """
    # Implementation notes:
    #   Typically, one would create the input ops and then the rest of the net.
    #   However, creating the input ops depends on loading the dataset, which
    #   can take a few minutes for COCO.
    #   We prefer to avoid waiting so debugging can fail fast.
    #   Thus, we create the net *without input ops* prior to loading the
    #   dataset, and then add the input ops after loading the dataset.
    #   Since we defer input op creation, we need to do a little bit of surgery
    #   to place the input ops at the start of the network op list.


def add_inference_inputs(model):
    """Create network input blobs used for inference."""

# ---------------------------------------------------------------------------- #
# Faster R-CNN models
# ---------------------------------------------------------------------------- #

def ResNet50_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_detection_model(
        model, ResNet.add_ResNet50_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )


def ResNet101_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_detection_model(
        model, ResNet.add_ResNet101_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )
