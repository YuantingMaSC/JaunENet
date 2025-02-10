from __future__ import absolute_import, division, print_function
import os
import random
from tensorflow.keras.applications.xception import Xception
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
# GPU settings
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0], "GPU")
import time
from prepare_data import load_and_preprocess_image,get_the_length_of_dataset
from parse_tfrecord import get_parsed_dataset
import math
from train import init_way

def generate_datasets(batch, dataset_dir):
    train_dir = dataset_dir+'train/'
    valid_dir = dataset_dir+'valid/'
    test_dir =  dataset_dir+'test/'
    train_tfrecord = dataset_dir + "train.tfrecord"
    valid_tfrecord = dataset_dir + "valid.tfrecord"
    test_tfrecord = dataset_dir + "test.tfrecord"
    print("getting training dataset")
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    print("getting valid dataset")
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    print("getting test dataset")
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)
    print(train_dataset)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)
    # read the dataset in the form of batch
    print('read the dataset in the form of batch')
    train_dataset = train_dataset.batch(batch_size=batch)
    valid_dataset = valid_dataset.batch(batch_size=batch)
    test_dataset = test_dataset.batch(batch_size=batch)

    return train_dataset, valid_dataset, test_dataset ,train_count, valid_count, test_count



def lr_decay(leaning_rate, epoch):
    if epoch <= 50:
        return leaning_rate
    elif epoch <= 100:
        return leaning_rate * 0.5
    elif epoch <= 300:
        return leaning_rate * 0.1
    else:
        return leaning_rate * 0.05
    # return leaning_rate


def process_features(features, data_augmentation): 
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()
    return images, labels


if __name__ == '__main__':
    # get the dataset
    dataset_dir = 'EDID/' #EDID_weakly_labeled
    init_way_ = 'pretain'

    IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128, 128, 3
    EPOCHS=500
    BATCH_SIZE=128
    save_every_n_epoch=20
    init_lr=2e-4
    NUM_CLASSES=10
    patience=80
    print("loading dataset...")
    train_dataset, valid_dataset, test_dataset ,train_count, valid_count, test_count = generate_datasets(batch = BATCH_SIZE, dataset_dir=dataset_dir)

    print('initiating model')
    # create model

    model = init_way(init_way_,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS,NUM_CLASSES)

    save_model_dir = f"saved_model_{init_way_}_{dataset_dir}/"
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    #define a most accurate model weights in validation
    valid_acc_sub = 0
    valid_acc_sub_weights = np.nan

    # start training
    patience_marker = 0
    for epoch in range(EPOCHS):
        if epoch>100:
            img_aug=True
        else:
            img_aug=False
        lr = lr_decay(init_lr,epoch)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        step = 0
        train_dataset.shuffle(100)  # 考虑训练集要shuffle一下，避免出现过拟合
        for features in train_dataset:
            step += 1
            X, labels = process_features(features, data_augmentation=img_aug)
            train_step(X, labels)
            print("\r","Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()),
                    end="",flush = True)

        for features in valid_dataset:
            valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{},"
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  EPOCHS,
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))

        if valid_accuracy.result().numpy() >= valid_acc_sub:
            valid_acc_sub = valid_accuracy.result().numpy()
            model.save_weights(filepath=save_model_dir + "best_valid_acc_model_weights.h5")
            patience_marker=0
        else:
            patience_marker+=1
            print(f"patience now step {patience_marker}")
        if patience_marker>patience:
            break
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}.h5".format(epoch))


    # save weights
    model.save_weights(filepath=save_model_dir+"model.h5")



    # # save the whole model
    # save_wholemodel_dir = "./saved_wholemodel"
    # tf.saved_model.save(model, save_wholemodel_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)

