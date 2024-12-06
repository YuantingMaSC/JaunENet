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
    import tensorflow as tf
    def add_last_layer(base_model, NUM_CLASSES):
        x = base_model.layers[-2].output
        # x = tf.keras.layers.Dense(512)(x)
        pred = tf.keras.layers.Dense(NUM_CLASSES)(x)
        pred = tf.nn.softmax(pred)
        model_new = tf.keras.Model(inputs=base_model.input, outputs=pred)
        return model_new

    # Create the Vision Transformer model
    # 创建模型
    vit_model = create_vit_model(
        image_size=128,  # CIFAR-10 图片大小
        num_classes=3,          # CIFAR-10 分类 
    )
    vit_model = add_last_layer(vit_model, 3)
    # 打印模型结构
    vit_model.summary()

    # 测试数据
    import numpy as np
    x_dummy = np.random.rand(8, 128, 128, 3).astype(np.float32)  # 8 张随机图像+
    # y_dummy = np.random.randint(0, 10, size=(8,))             # 对应 8 个标签

    # 编译并训练模型
    # vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # vit_model.fit(x_dummy, y_dummy, epochs=1)

    print(vit_model(x_dummy).shape)
