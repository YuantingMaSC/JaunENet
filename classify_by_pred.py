import os
import shutil

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import pandas as np
import tensorflow as tf
# GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from configuration import IMAGE_WIDTH,IMAGE_HEIGHT,CHANNELS
from train import get_model
from prepare_data import load_and_preprocess_image

def get_class_id(image_root):
    id_cls = {}
    for i, item in enumerate(os.listdir(image_root)):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    return id_cls

if __name__ == '__main__':

    model = get_model()
    weight_file_name = 'top_valid_acc_model_weights_0.977.h5'
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    model.load_weights(filepath=save_model_dir+weight_file_name)
    converteddir ='./dicom_to_jpg'
    filenames = os.listdir(converteddir)
    print(filenames)
    for filename in filenames:
        image_raw = tf.io.read_file(converteddir+'/'+filename)
        image_tensor = load_and_preprocess_image(image_raw,data_augmentation=False)
        # plt.imshow((image_tensor))
        # plt.show()
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        pred = model(image_tensor, training=False)

        idx = tf.math.argmax(pred, axis=-1).numpy()[0]
        id_cls = get_class_id("./original_dataset")
        img_class = id_cls[idx]
        if not os.path.exists('./pred_class/'):
            os.mkdir('./pred_class/')
        category_path = './pred_class/' + img_class
        if not os.path.exists(category_path):
            os.mkdir(category_path)
        image_processed_name = category_path+'/'+filename
        tf.keras.utils.save_img(image_processed_name, tf.squeeze(image_tensor).numpy())
        # shutil.copy(converteddir+'/'+filename, category_path+'/'+filename)
        print("The predicted category of this picture is: {}".format(img_class))