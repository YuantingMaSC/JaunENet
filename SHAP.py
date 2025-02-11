# -*- coding: utf-8 -*-'''
'''
Author       : Yuanting Ma
Github       : https://github.com/YuantingMaSC
LastEditors  : Yuanting_Ma 
Date         : 2024-12-06 09:23:59
LastEditTime : 2025-02-11 10:30:41
FilePath     : /JaunENet/SHAP.py
Description  : 
Copyright (c) 2025 by Yuanting_Ma@163.com, All Rights Reserved. 
'''
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import shap
from prepare_data import load_and_preprocess_image,generate_datasets_shap
from train import init_way
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # whether to use GPU
import numpy as np

BATCH_SIZE = 10

def get_class_id(image_root):
    id_cls = {}
    for i, item in enumerate(os.listdir(image_root)):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    return id_cls

def process_features(features, data_augmentation,image_show = False):  
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation,image_show = image_show)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()
    meta = 2*(features['meta'].numpy() / np.array([1,1,100,1,1,1,1,1,1])) - 1 # meta_data -> [-1,1]
    return images, meta, labels


tf.device('/cpu:0')
init_way_ = 'EDID_weakly_labeled'
IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128,128,3
model = init_way(init_way_,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, 3, show_summary=False)
model.load_weights("saved_model_EDID_weakly_labeled/best_valid_acc_model_weights.h5")
# print(model.summary())


train_dataset = generate_datasets_shap(batch=1)
images = []
step = 1
num = 600
for feature in train_dataset:
    image_raw  = feature['image_raw'].numpy()
    for image in image_raw:
        image_tensor = tf.io.decode_image(contents=image, channels=CHANNELS, dtype=tf.dtypes.float32)
        hight = min(image_tensor.shape[0], image_tensor.shape[1])
        image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, target_height=hight,
                                                        target_width=hight) # crop the irregular image to a square
        image_tensor = tf.image.resize(image_tensor,size=(IMAGE_HEIGHT,IMAGE_WIDTH))
        image_np = image_tensor.numpy()
        images.append(image_np)

    print("\r" + "loading backgrounds..." + "█" * int(step / num * 30) + '%.2f' % (step / num * 100) + "%", end="",
          flush=True)
    if step >= num:
        break
    step += 1
images = np.array(images)
print('\n', images.shape, type(images))

class_names = get_class_id('./dataset/train')

print("type bks", type(images))
explainer = shap.GradientExplainer(model, images)

# we explain the model's predictions on the first three samples of the test set
X_dir = './shap_image_draw/'
if not os.path.exists(X_dir):
    os.mkdir(X_dir)
X = []
for name in os.listdir(X_dir):
    image_raw = tf.io.read_file(X_dir+name)
    image_processed = load_and_preprocess_image(image_raw, data_augmentation=False, image_show=False)
    X.append(image_processed.numpy())

X = np.array(X)
print(X.shape,type(X))
shap_values, indexes = explainer.shap_values(X, ranked_outputs=3)

index_names = np.vectorize(lambda x: class_names[x])(indexes)

shap.image_plot(shap_values, X,index_names)