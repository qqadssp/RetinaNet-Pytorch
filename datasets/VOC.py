from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import torch
import torchvision.transforms as transforms

from PIL import Image
from utils.config import cfg
from datasets.encoder import DataEncoder
from datasets.transform import resize, random_flip, random_crop, center_crop

import xml.etree.ElementTree as ET

class VOCdataset(torch.utils.data.Dataset):
    def __init__(self):
        self.category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                         'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.img_root_dir = os.path.join(cfg.TRAIN.DATASETS_DIR, cfg.TRAIN.DATASETS[0], 'JPEGImages')
        self.img_list = os.path.join(cfg.TRAIN.DATASETS_DIR, cfg.TRAIN.DATASETS[0], 'ImageSets', 'Main', cfg.TRAIN.DATASETS[1])
        self.annotations_dir = os.path.join(cfg.TRAIN.DATASETS_DIR, cfg.TRAIN.DATASETS[0], 'Annotations')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
        self.train = True
        self.input_size = cfg.TRAIN.SCALES

        self.fnames = []
        self.boxes = []
        self.labels = []
        self.num_samples = 0
        self.get_img_annotations()

        self.encoder = DataEncoder(self.input_size)

    def get_img_annotations(self): 
        with open(self.img_list) as f:
            lines = f.readlines()
        self.num_samples = len(lines)

        for line in lines:
            splited = line.strip()
            self.fnames.append(splited + '.jpg')
            box = []
            label = []
            ann = os.path.join(self.annotations_dir, splited+'.xml')
            rec = parse_rec(ann)
            for r in rec : 
                box.append(r['bbox'])
                label.append(self.category.index(r['name']))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.img_root_dir, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, size)
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, size)

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = self.input_size[1]
        w = self.input_size[0]
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

# test()
