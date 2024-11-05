#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import torch
import h5py
import numpy as np
#import tensorflow as tf
#from keras.preprocessing import image
import cv2
import sys
sys.path.append("..")
# from networks.resnet import ResNet18
# from torchvision.models import resnet18
from mnist import MNISTNet()

def dump_image(x, filename, format):
    #img = image.array_to_img(x, scale=False)
    #img.save(filename, format)
    #return
    cv2.imwrite(filename, x)


def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            print("h5py keys: ", keys)
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset

def load_model(model_file, device):
    # use_cuda = torch.cuda.is_available()
    print("In load model")
    net = MNISTNet().to(device)
    model = torch.load(model_file,  map_location=torch.device('cpu')) 
    net.load_state_dict(model)
    print("model done")
    return net
