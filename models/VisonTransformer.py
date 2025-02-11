# -*- coding: utf-8 -*-'''
'''
Author       : Yuanting Ma
Github       : https://github.com/YuantingMaSC
LastEditors  : Yuanting_Ma 
Date         : 2024-12-06 09:19:02
LastEditTime : 2025-02-11 10:22:18
FilePath     : /JaunENet/models/VisonTransformer.py
Description  : 
Copyright (c) 2025 by Yuanting_Ma@163.com, All Rights Reserved. 
'''
from vit_keras import vit, utils

def create_vit_model(image_size=128,  num_classes=3):
    model = vit.vit_b16(
        image_size=image_size,
        activation='sigmoid',
        pretrained=False,
        include_top=True,
        pretrained_top=False,
        classes=num_classes,
    )
    return model

if __name__ == '__main__':
    from tensorflow.keras.datasets import cifar10
    import numpy as np
    import tensorflow as tf
    def add_last_layer(base_model, NUM_CLASSES):
        x = base_model.layers[-2].output
        # x = tf.keras.layers.Dense(512)(x)
        pred = tf.keras.layers.Dense(NUM_CLASSES)(x)
        pred = tf.nn.softmax(pred)
        model_new = tf.keras.Model(inputs=base_model.input, outputs=pred)
        return model_new

    # Create the Vision Transformer model
    # Create model
    vit_model = create_vit_model(
        image_size=128,  # CIFAR-10 image size
        num_classes=3,   # CIFAR-10 classes
    )
    vit_model = add_last_layer(vit_model, 3)
    # Print model structure
    vit_model.summary()

    # Test data

    x_dummy = np.random.rand(8, 128, 128, 3).astype(np.float32)  # 8 random images
    # y_dummy = np.random.randint(0, 10, size=(8,))  # 8 tags

    # Compile and train the model
    # vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # vit_model.fit(x_dummy, y_dummy, epochs=1)

    print(vit_model(x_dummy).shape)
