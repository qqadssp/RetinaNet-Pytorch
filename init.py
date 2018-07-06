'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from modeling.ResNet_FPN import FPN50
from modeling.RetinaNet import RetinaNet

def main(): 
    """
    """
    if not os.path.exists('./checkpoint/resnet50-19c8e357.pth'): 
        print('Pretrained model does not exist')
        print('Download from: https://download.pytorch.org/models/resnet50-19c8e357.pth')
        sys.exit(1)

    print('Loading pretrained ResNet50 model..')
    d = torch.load('./checkpoint/resnet50-19c8e357.pth')

    print('Loading into FPN50..')
    fpn = FPN50()
    dd = fpn.state_dict()
    for k in d.keys():
        if not k.startswith('fc'):  # skip fc layers
            dd[k] = d[k]

    print('Saving RetinaNet..')
    net = RetinaNet(FPN50, 20)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    pi = 0.01
    init.constant_(net.cls_head[-1].bias, -math.log((1-pi)/pi))

    net.fpn.load_state_dict(dd)
    torch.save(net.state_dict(), './checkpoint/init.pth')
    print('Done!')

if __name__ == '__main__': 
    main()
