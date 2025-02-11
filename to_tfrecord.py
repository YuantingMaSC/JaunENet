# -*- coding: utf-8 -*-'''
'''
Author       : Yuanting Ma
Github       : https://github.com/YuantingMaSC
LastEditors  : Yuanting_Ma 
Date         : 2024-12-06 09:23:59
LastEditTime : 2025-02-11 10:33:30
FilePath     : /JaunENet/to_tfrecord.py
Description  : 
Copyright (c) 2025 by Yuanting_Ma@163.com, All Rights Reserved. 
'''
import pandas as pd
import os
import tensorflow as tf
from prepare_data import get_images_and_labels
import random

dataset_dir = 'dataset/'
train_dir = dataset_dir+'train/'
valid_dir = dataset_dir+'valid/'
test_dir =  dataset_dir+'test/'
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"

# convert a value to a type compatible tf.train.Feature
def _bytes_feature(value):
    # Returns a bytes_list from a string / byte.
    if isinstance(value, type(tf.constant(0.))):  # Check if the type of value is the same as tf.constant()
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # Returns a float_list from a float / double.
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    # Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def image_example(image_string,meta, label):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
        'meta' : _float_feature(meta)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def shuffle_dict(original_dict):
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict


def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    image_paths, image_labels = get_images_and_labels(dataset_dir)
    image_paths_and_labels_dict = {}
    for i in range(len(image_paths)):
        image_paths_and_labels_dict[image_paths[i]] = image_labels[i]
    # shuffle the dict
    image_paths_and_labels_dict = shuffle_dict(image_paths_and_labels_dict)
    # write the images and labels to tfrecord format file
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        for image_path, label in image_paths_and_labels_dict.items():
            # print("Writing to tfrecord: {}".format(image_path))
            image_string = open(image_path, 'rb').read()
            # metai = meta.loc[image_name[:num]].tolist()
            metai = [0,0,0,0,0,0,0,0,0]
            tf_example = image_example(image_string, metai, label)
            writer.write(tf_example.SerializeToString())


def to_tf_record(mode):
    print("writing to training tf_record...")
    dataset_to_tfrecord(dataset_dir=train_dir, tfrecord_name=train_tfrecord)
    print("writing to valid tf_record...")
    dataset_to_tfrecord(dataset_dir=valid_dir, tfrecord_name=valid_tfrecord)
    print("writing to test tf_record...")
    dataset_to_tfrecord(dataset_dir=test_dir, tfrecord_name=test_tfrecord)


if __name__ == '__main__':
    to_tf_record(mode="normal")