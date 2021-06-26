from __future__ import print_function, absolute_import
import torch
import torch.utils.data as data
import os
import numpy as np
import cv2

def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

class _OWN(data.Dataset):
    def __init__(self, config, is_train=True):
        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']
        
        init_img=[]
        init_text=[]

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            for c in file.readlines():
                init_img.append(c.split(' ')[0])
                init_text.append(c.split(' ')[-1][:-1])

        sorted_index = len_argsort(init_text)
        init_img = [init_img[i] for i in sorted_index]
        init_text = [init_text[i] for i in sorted_index]
        self.labels = [{init_img[i]: init_text[i]} for i,_ in enumerate(init_img) ]

        if self.is_train:
            print("load {} train images!".format(self.__len__()))
        else:
            print("load {} test images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape
        img = cv2.resize(img, (0, 0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img / 255. - self.mean) / self.std

        img = img.transpose([2, 0, 1])

        return img, idx

def collate_fn(batch):
    idxs = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        idxs.append(sample[1])
    return torch.Tensor(imgs), idxs
""" 
import yaml
from easydict import EasyDict as edict
from  alphabets import alphabet
from torch.utils.data import DataLoader
import utils

def parse_arg():
    cfg = "config/config.yaml"
    with open(cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    print(len(config.DATASET.ALPHABETS))
    return config

if __name__ == '__main__':
    config = parse_arg()
    train_dataset = _OWN(config, is_train=True)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    
    #47834, 181173, 42414, 35567
    idxs=[0,1,2,3,4,5,6,7,8,9,10]
    #.values()
    for idx in idxs:
        print(train_dataset.labels[idx])
    
    for i, (inp, idx) in enumerate(train_loader):
        
        labels = utils.get_batch_label(train_dataset, idx)
        
        inp = inp.to(device)
        batch_size = inp.size(0)
        text, length = converter.encode(labels)
        print("----------------")
        print(i,":",idx)
        print("label:",labels)
        print("text:",text)
        print("length:",length)
    
"""