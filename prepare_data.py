import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import numpy as np
from parse_tfrecord import get_parsed_dataset

IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128, 128, 3
dataset_dir = 'dataset/'
train_dir = dataset_dir+'train/'
valid_dir = dataset_dir+'valid/'
test_dir =  dataset_dir+'test/'
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"

def load_and_preprocess_image(image_raw,  IMAGE_HEIGHT=IMAGE_HEIGHT, IMAGE_WIDTH=IMAGE_WIDTH, CHANNELS=CHANNELS, data_augmentation=False,image_show = False): #数据增强部分代码需要替换
    # decode tfa_tensor-> (n,268,268,3),tf_tensor->(268,268,3)
    zoom_rate = 1.05
    image_tensor = tf.io.decode_image(contents=image_raw, channels=CHANNELS, dtype=tf.dtypes.float32)
    if image_show:
        plt.imshow(image_tensor)
        plt.show()
        plt.clf()
    hight = min(image_tensor.shape[0], image_tensor.shape[1])
    image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, target_height=hight,
                                                    target_width=hight)  # 各年份图像分辨率不一致，先crop成原图短边长的方形图
    if hight<IMAGE_WIDTH:
        image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, target_height=int(IMAGE_WIDTH * zoom_rate),
                                                    target_width=int(IMAGE_WIDTH * zoom_rate))  # 低分辨率图像padding
    else:
        image_tensor = tf.image.resize(image_tensor, size=(int(IMAGE_WIDTH * zoom_rate), int(IMAGE_WIDTH * zoom_rate)))#高分辨率图像直接resize采样
    if data_augmentation:
        image = tf.reshape(tf.cast(image_tensor, dtype=tf.float32),
                           shape=(1, int(IMAGE_WIDTH * zoom_rate), int(IMAGE_WIDTH * zoom_rate), 3)) #tfa 输入的是一个四维的tensor
        if np.random.rand() < 0.05:  # 以0.8的概率来旋转
            image = tfa.image.rotate(images=image, angles=0.1 * np.random.rand() * np.pi)
        if np.random.rand() < 0.05:  # 以0.8的概率来cutout
            image = tfa.image.random_cutout(images=image, mask_size=(int(IMAGE_WIDTH * 0.10), int(IMAGE_WIDTH * 0.10)))  # 随机裁剪
        image = tf.squeeze(image)
        if np.random.rand() < 0.05:
            image = tf.image.random_flip_left_right(image=image)  # 1:2的概率水平翻转
        if np.random.rand() < 0.05:
            image = tf.image.transpose(image)
        # image = tf.add(image, tf.cast(
        #     tf.random.normal(shape=[int(IMAGE_WIDTH * zoom_rate), int(IMAGE_WIDTH * zoom_rate), 3], mean=0, stddev=0.02),
        #     tf.float32))
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.8)  # 随机设置图片的对比度
        # # image = tf.image.random_hue(image, max_delta=0.3)  # 随机设置图片的色度
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.8)  # 随机设置图片的饱和度
        if np.random.rand() < 0.05:
            image = tf.image.random_brightness(image=image, max_delta=0.05)
        image = tf.image.random_crop(value=image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        # 
    else:
        image = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    if image_show:
        plt.imshow(image)
        plt.show()
        plt.clf()
    return image


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count


def generate_datasets(batch):
    print("getting training dataset")
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    print("getting valid dataset")
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    print("getting test dataset")
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)
    # read the dataset in the form of batch
    print('read the dataset in the form of batch')
    train_dataset = train_dataset.batch(batch_size=batch)
    valid_dataset = valid_dataset.batch(batch_size=batch)
    test_dataset = test_dataset.batch(batch_size=batch)

    return train_dataset, valid_dataset, test_dataset ,train_count, valid_count, test_count

def generate_datasets_shap(batch):
    print("getting training dataset")
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    train_dataset = train_dataset.batch(batch_size=batch)
    return train_dataset