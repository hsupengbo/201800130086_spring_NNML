# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from ..config import IMAGE_MEAN
from ..ctpn_utils import cal_rpn


def read_xml(path):
    gt_boxes = []
    img_file = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            img_file = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))

                    gt_boxes.append((xmin, ymin, xmax, ymax))

    return np.array(gt_boxes), img_file


# for ctpn text detection
class VOCDataset(Dataset):
    def __init__(self,
                 data_dir,
                 labels_dir):
        """
        param txt_file: image name list text file
        param data_dir: image's directory
        param labels_dir: annotations' directory
        """
        if not os.path.isdir(data_dir):
            raise Exception('[ERROR] {} is not a directory'.format(data_dir))
        if not os.path.isdir(labels_dir):
            raise Exception('[ERROR] {} is not a directory'.format(labels_dir))

        self.data_dir = data_dir
        self.img_names = os.listdir(self.data_dir)
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        print(img_path)
        xml_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.xml'))
        gt_box, _ = read_xml(xml_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            new_x1 = w - gt_box[:, 2] - 1
            new_x2 = w - gt_box[:, 0] - 1
            gt_box[:, 0] = new_x1
            gt_box[:, 2] = new_x2

        [cls, regrow], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gt_box)

        m_img = img - IMAGE_MEAN

        regrow = np.hstack([cls.reshape(cls.shape[0], 1), regrow])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regrow = torch.from_numpy(regrow).float()

        return m_img, cls, regrow
