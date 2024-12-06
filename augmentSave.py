import os
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prepare_data import load_and_preprocess_image

aug_num_sub = 1
IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS = 200,200,3
task_list = ['EDID'] #,'ODIR5k_weakly_labeled', EDID_weakly_labeled
for task in task_list:
    trainset_dir = f'or_{task}'
    for set_type in os.listdir(trainset_dir):
        image_list = {}
        print(set_type)
        #建立sugset文件夹
        augset_dir = f'{task}/{set_type}'

        if not os.path.exists(augset_dir):
            os.makedirs(augset_dir)

        out_list = os.listdir(trainset_dir+'/'+set_type)
        for item in out_list:
            inner_dir = trainset_dir +'/'+set_type + '/'+ item
            class_list = os.listdir(inner_dir)
            image_list[item] = class_list
        # print(image_list)
        for diag in image_list:
            _list = image_list[diag]
            _images_num = len(_list)
            print("\nclass",diag,' has ',_images_num,' samples ')
            repeat_num = 1

            class_dir = augset_dir+'/'+diag
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
            total = repeat_num*len(_list)
            step = 1
            # for num in range(repeat_num):
            for image_name in _list:
                _image_path = trainset_dir +'/'+ set_type +'/'+ diag +'/'+image_name
                print(_image_path)
                image_raw = tf.io.read_file(_image_path)
                image_processed = load_and_preprocess_image(image_raw, data_augmentation=False,image_show=False)
                image_name_head = os.path.splitext(image_name)[0]
                image_processed_name = class_dir +'/'+image_name_head +'_'+"%05d" +'copy'+'.jpg'
                tf.keras.utils.save_img(image_processed_name,image_processed.numpy())
                step += 1
                print('\r','class {1} {0:.1f} % has been augmented'.format(step/total*100,diag),end ='',flush = True)