# -*- coding: utf-8 -*-'''
'''
Author       : Yuanting Ma
Github       : https://github.com/YuantingMaSC
LastEditors  : Yuanting_Ma 
Date         : 2024-12-06 09:23:59
LastEditTime : 2025-02-11 10:20:05
FilePath     : /JaunENet/configuration.py
Description  : 
Copyright (c) 2025 by Yuanting_Ma@163.com, All Rights Reserved. 
'''
DEVICE = "gpu"   # cpu or gpu
import tensorflow as tf
# some training parameters

init_lr = 3e-5
EPOCHS = 500
save_every_n_epoch = 25
BATCH_SIZE = 128
NUM_CLASSES = 3
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3

metasize = 9


dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"


