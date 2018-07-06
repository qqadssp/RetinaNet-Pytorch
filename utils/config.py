"""Config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See configs/*/*.yaml for example config files

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from future.utils import iteritems
from past.builtins import basestring
import copy
import logging
import numpy as np
import os
import os.path as osp
import yaml

from utils.collection import AttrDict

logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
cfg = __C

# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Initialize network with weights from this .pkl file
__C.TRAIN.WEIGHTS = b''

# Datasets to train on
# Available dataset list: detectron.datasets.dataset_catalog.datasets()
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

__C.TRAIN.DATASETS_DIR = ' '
__C.TRAIN.SAVE_CHECKPOINTS_EPOCHS = 10
__C.TRAIN.MAX_EPOCHS = 100
__C.TRAIN.BATCH_SIZE = 8
# Scales to use during training
# Each scale is the pixel size of an image's shortest side
# If multiple scales are listed, then one is selected uniformly at random for
# each training image (i.e., scale jitter data augmentation)
__C.TRAIN.SCALES = (800, )

# Training will resume from the latest snapshot (model checkpoint) found in the
# output directory
__C.TRAIN.AUTO_RESUME = False

__C.TRAIN.RESUME_FILE = 'RESUME.PTH'
# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Initialize network with weights from this .pkl file
__C.TEST.WEIGHTS = b''

# Datasets to test on
# Available dataset list: detectron.datasets.dataset_catalog.datasets()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASETS = ()

__C.TEST.DATASETS_DIR = ' '
# Scale to use during testing
__C.TEST.SCALES = (600,)

__C.TEST.OUTPUT_DIR = b''

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The type of model to use
# The string must match a function in the modeling.model_builder module
# (e.g., 'generalized_rcnn', 'mask_rcnn', ...)
__C.MODEL.TYPE = b''

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
__C.MODEL.CONV_BODY = b''

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1

# ---------------------------------------------------------------------------- #
# RetinaNet options
# ---------------------------------------------------------------------------- #
__C.RETINANET = AttrDict()

# RetinaNet is used (instead of Fast/er/Mask R-CNN/R-FCN/RPN) if True
__C.RETINANET.RETINANET_ON = False

# Anchor aspect ratios to use
__C.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)

# At each FPN level, we generate anchors based on their scale, aspect_ratio,
# stride of the level, and we multiply the resulting anchor by ANCHOR_SCALE
__C.RETINANET.ANCHOR_SCALE = 4

__C.RETINANET.SCALE_RATIOS = (1.0, pow(2.0, 1.0/3.0), pow(2.0, 2.0/3.0))
# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
__C.RETINANET.NUM_CONVS = 4

# Focal loss parameter: alpha
__C.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
__C.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
__C.RETINANET.PRIOR_PROB = 0.01

# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

__C.SOLVER.OPTIMIZER = 'sgd'

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITER = 40000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
__C.FPN = AttrDict()

# FPN is enabled if True
__C.FPN.FPN_ON = False

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# ---------------------------------------------------------------------------- #
# ResNets options ("ResNets" = ResNet and ResNeXt)
# ---------------------------------------------------------------------------- #
__C.RESNETS = AttrDict()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
__C.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
__C.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
__C.RESNETS.STRIDE_1X1 = True

# Residual transformation function
__C.RESNETS.TRANS_FUNC = b'bottleneck_transformation'
# ResNet's stem function (conv1 and pool1)
__C.RESNETS.STEM_FUNC = b'basic_bn_stem'
# ResNet's shortcut function
__C.RESNETS.SHORTCUT_FUNC = b'basic_bn_shortcut'

# Apply dilation in stage "res5"
__C.RESNETS.RES5_DILATION = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# "Fun" fact: the history of where these values comes from is lost
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# Output basedir
__C.OUTPUT_DIR = b'/tmp'

# Dump detection visualizations
__C.VIS = False

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set(
    {
        'FINAL_MSG',
        'MODEL.DILATION',
        'ROOT_GPU_ID',
        'RPN.ON',
        'TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED',
        'TRAIN.DROPOUT',
        'USE_GPU_NMS',
        'TEST.NUM_TEST_IMAGES',
    }
)

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
}


# ---------------------------------------------------------------------------- #
# Renamed modules
# If a module containing a data structure used in the config (e.g. AttrDict)
# is renamed/moved and you don't want to break loading of existing yaml configs
# (e.g. from weights files) you can specify the renamed module below.
# ---------------------------------------------------------------------------- #
_RENAMED_MODULES = {
    'EXAMPLE.RENAMED.MODULE': 'EXAMPLE.MODULE',  # Dummy example to follow    
}


def assert_and_infer_cfg(cache_urls=True, make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.RPN_ON = True
    if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
        __C.TEST.PRECOMPUTED_PROPOSALS = False
    if cache_urls:
        cache_cfg_urls()
    if make_immutable:
        cfg.immutable(True)


def cache_cfg_urls():
    """Download URLs in the config, cache them locally, and rewrite cfg to make
    use of the locally cached file.
    """
    __C.TRAIN.WEIGHTS = cache_url(__C.TRAIN.WEIGHTS, __C.DOWNLOAD_CACHE)
    __C.TEST.WEIGHTS = cache_url(__C.TEST.WEIGHTS, __C.DOWNLOAD_CACHE)
    __C.TRAIN.PROPOSAL_FILES = tuple(
        cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TRAIN.PROPOSAL_FILES
    )
    __C.TEST.PROPOSAL_FILES = tuple(
        cache_url(f, __C.DOWNLOAD_CACHE) for f in __C.TEST.PROPOSAL_FILES
    )


def get_output_dir(datasets, training=True):
    """Get the output directory determined by the current global config."""
    assert isinstance(datasets, (tuple, list, basestring)), \
        'datasets argument must be of type tuple, list or string'
    is_string = isinstance(datasets, basestring)
    dataset_name = datasets if is_string else ':'.join(datasets)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset-name>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset_name, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir

def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v

def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        logger.warn(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
