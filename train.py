from __future__ import absolute_import, division, print_function
import os
import random
from tensorflow.keras.applications.xception import Xception
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0] 
    tf.config.experimental.set_memory_growth(gpu0, True) 
    tf.config.set_visible_devices([gpu0], "GPU")
import time
from prepare_data import generate_datasets
from prepare_data import load_and_preprocess_image
import math
from tensorflow.keras.applications import efficientnet,densenet,VGG19, ResNet50V2, InceptionV3, InceptionResNetV2, \
    MobileNetV3Large
from models.ConvNeXt import ConvNeXtLarge
from models.VisonTransformer import create_vit_model

def setup_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    tf.random.set_seed(seed)  

def init_way(init_weight, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS,NUM_CLASSES, show_summary=True):
    freeze_num =0
    if init_weight == "Imagenet":
        model = get_model('imagenet',1000,224,224)
        freeze_num = 5 
    elif init_weight=='EDID_weakly_labeled':
        model = get_model(None,2,128,128)
        print("loading the previous weights")
        model.load_weights("saved_model_pretain_EDID_weakly_labeled/best_valid_acc_model_weights.h5")
        freeze_num = 5  
    elif init_weight=='EDID':
        model = get_model(None,10,128,128)
        print("loading the previous weights")
        model.load_weights("saved_model_pretain_EDID/best_valid_acc_model_weights.h5")
        freeze_num = 5  
    elif init_weight == "ISIC":
        model = get_model(None,1000,128,128)
        print("loading the previous weights")
        model.load_weights('./saved_model_pretain_ISIC/' + "skin_data_weights.h5")
        freeze_num = 5  
    elif init_weight == 'ISIC_weakly_labeled':
        model = get_model(None,1000,128,128)
        print("loading the previous weights")
        model.load_weights('./saved_model_pretain_ISIC_weakly_labeled/' + 'skin_data_unlabled.h5')
        freeze_num = 5  

    elif init_weight == 'JaunENet':
        model = get_model(None, 1000, 128, 128)
    elif init_weight == 'Densenet':
        model = densenet.DenseNet121(weights=None,input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS),pooling='avg',
                                     classes=NUM_CLASSES)
    elif init_weight == 'VGG':
        model = VGG19(weights=None,input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS),pooling='avg',classes=NUM_CLASSES,
                      classifier_activation='softmax')
    elif init_weight =='ConvNeXt':
        model=ConvNeXtLarge(weights=None,model_name='convnext_large',include_top=True,
                            input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS), classes=NUM_CLASSES,classifier_activation='softmax')
    elif init_weight == 'Resnet':
        model = ResNet50V2(weights=None, input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS),pooling='avg',
                           classes=NUM_CLASSES,classifier_activation='softmax')
    elif init_weight == 'Inception':
        model = InceptionV3(weights=None, input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS),pooling='avg',
                            classes=NUM_CLASSES,classifier_activation='softmax')

    elif init_weight == 'Mobilenet':
        model = MobileNetV3Large(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS), alpha=1.0, include_top=True, weights=None,
                            pooling='avg', classes=NUM_CLASSES,classifier_activation='softmax')
    elif init_weight == 'Xception':
        model = Xception(weights=None, input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS),pooling='avg',
                                  classes=NUM_CLASSES,classifier_activation='softmax')
    elif init_weight == 'Vit':
        model = create_vit_model(
                    image_size=128,
                    num_classes=3,
                )
        return model
    if init_weight == 'pretain':
        model = get_model(None,NUM_CLASSES,128,128)
        model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        model.summary()
        return model
    else:
        raise ValueError("initial way name error !")
    model = add_last_layer(model,NUM_CLASSES)
    model = freeze(model,freeze_num)
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    if show_summary:
        model.summary()
    return model


def get_model(init_weights,original_class_num,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS=3):
    return efficientnet.EfficientNetB0(
        include_top=True, weights=init_weights, input_tensor=None,
        input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH, CHANNELS), pooling='avg', classes=original_class_num,
        classifier_activation='softmax')

def add_last_layer(base_model, NUM_CLASSES):
    x = base_model.layers[-2].output
    pred = tf.keras.layers.Dense(NUM_CLASSES)(x)
    pred = tf.nn.softmax(pred)
    model_new = tf.keras.Model(inputs=base_model.input, outputs=pred)
    return model_new

def freeze(model_,freeze_num):
    for layer in model_.layers[:freeze_num]:
        layer.trainable = False
    return model_

def lr_decay(leaning_rate, epoch):
    if epoch <= 50:
        return leaning_rate
    elif epoch <= 100:
        return leaning_rate * 0.5
    elif epoch <= 300:
        return leaning_rate * 0.1
    else:
        return leaning_rate * 0.05

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
    IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128, 128, 3
    EPOCHS=500
    BATCH_SIZE=48
    save_every_n_epoch=20
    init_lr=2e-5
    NUM_CLASSES=3
    patience=80
    print("loading dataset...")
    train_dataset, valid_dataset, test_dataset ,train_count, valid_count, test_count = generate_datasets(batch = BATCH_SIZE)

    print('initiating model')
    # create model
    init_way_ = 'JaunENet'  
    model = init_way(init_way_,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_CLASSES)


    save_model_dir = "saved_model_{}/".format(init_way_)
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
    valid_acc_sub = 0.4
    valid_acc_sub_weights = np.nan
    
    # start training
    patience_marker = 0
    for epoch in range(EPOCHS):
        if epoch>50 :
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

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        #保存目前为止valid_acc最高的model weights
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

