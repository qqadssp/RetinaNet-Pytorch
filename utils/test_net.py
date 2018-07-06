from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import sys

import torch
import torchvision.transforms as transforms

from PIL import Image
from utils.config import cfg
from modeling.model_builder import create
from datasets.encoder import DataEncoder
from datasets.transform import resize

category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def test_model():
    """Model testing loop."""
    logger = logging.getLogger(__name__)

    model = create(cfg.MODEL.TYPE, cfg.MODEL.CONV_BODY, cfg.MODEL.NUM_CLASSES)
    checkpoint = torch.load(os.path.join('checkpoint', cfg.TEST.WEIGHTS))
    model.load_state_dict(checkpoint['net'])


    if not torch.cuda.is_available(): 
        logger.info('cuda not find')
        sys.exit(1)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

    img_dir = os.path.join(cfg.TEST.DATASETS_DIR, cfg.TEST.DATASETS[0], 'JPEGImages')
    img_list = os.path.join(cfg.TEST.DATASETS_DIR, cfg.TEST.DATASETS[0], 'ImageSets', 'Main', cfg.TEST.DATASETS[1])

    with open(img_list, 'r') as lst : 
        img_list = lst.readlines()
    img_nums = len(img_list)

    test_scales = cfg.TEST.SCALES
    dic = {}
    for i in range(20) : 
        dic[str(i)] = []

    for im in xrange(img_nums):
        if im % 100 == 0 : logger.info('{} imgs were processed, total {}'. format(im, img_nums))
        img = Image.open(os.path.join(img_dir, img_list[im].strip() + '.jpg'))
        img_size = img.size
        img = img.resize(test_scales)

        x = transform(img)
        x = x.unsqueeze(0)
        x = torch.autograd.Variable(x)
        loc_preds, cls_preds = model(x)

        loc_preds = loc_preds.data.squeeze().type(torch.FloatTensor)
        cls_preds = cls_preds.data.squeeze().type(torch.FloatTensor)

        encoder = DataEncoder(test_scales)
        boxes, labels, sco, is_found = encoder.decode(loc_preds, cls_preds, test_scales)
        if is_found :
            img, boxes = resize(img, boxes, img_size)

            boxes = boxes.ceil()
            xmin = boxes[:, 0].clamp(min = 1)
            ymin = boxes[:, 1].clamp(min = 1)
            xmax = boxes[:, 2].clamp(max = img_size[0] - 1)
            ymax = boxes[:, 3].clamp(max = img_size[1] - 1)

            nums = len(boxes)
            for i in range(nums) : 
                dic[str(labels[i].item())].append([img_list[im].strip(), sco[i].item(), xmin[i].item(), ymin[i].item(), xmax[i].item(), ymax[i].item()])
    
    for key in dic.keys() : 
        logger.info('category id: {}, category name: {}'. format(key, category[int(key)]))
        file_name = cfg.TEST.OUTPUT_DIR + 'comp4_det_test_'+category[int(key)]+'.txt'
        with open(file_name, 'w') as comp4 :  
            nums = len(dic[key])
            for i in range(nums) : 
                img, cls_preds, xmin, ymin, xmax, ymax = dic[key][i]
                if cls_preds > 0.5 : 
                    cls_preds = '%.6f' % cls_preds
                    loc_preds = '%.6f %.6f %.6f %.6f' % (xmin, ymin, xmax, ymax)
                    rlt = '{} {} {}\n'.format(img, cls_preds, loc_preds)
                    comp4.write(rlt)
