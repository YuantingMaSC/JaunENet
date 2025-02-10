import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model

MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "default_size": 224,
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "default_size": 224,
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "default_size": 224,
    },
}

class LayerScale(layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(self.init_values * tf.ones((self.projection_dim,)))

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim
            }
        )
        return config
    

# 随机深度模块
'''
 drop_path_rate (float)：丢弃路径的概率。应该在[0, 1]。
 返回：残差路径丢弃或保留的张量。
'''
class StochasticDepth(layers.Layer):
    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config
    
def ConvNextBlock(inputs,
                  projection_dim,  # 卷积层的filters数量
                  drop_path_rate=0.0,  # 丢弃路径的概率。
                  layer_scale_init_value=1e-6,
                  name=None):
    x = inputs
    # Depthwise卷积是分组卷积的一种特殊情况：当分组数=通道数
    x = layers.Conv2D(filters=projection_dim,
                      kernel_size=(7, 7),
                      padding='same',
                      groups=projection_dim,
                      name=name + '_depthwise_conv')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=name + '_layernorm')(x)
    x = layers.Dense(4 * projection_dim, name=name + '_pointwise_conv_1')(x)
    x = layers.Activation('gelu', name=name + '_gelu')(x)
    x = layers.Dense(projection_dim, name=name + '_pointwise_conv_2')(x)

    if layer_scale_init_value is not None:
        # Layer scale module
        x = LayerScale(layer_scale_init_value, projection_dim, name=name + '_layer_scale')(x)
    if drop_path_rate:
        # 随机深度模块
        layer = StochasticDepth(drop_path_rate, name=name + '_stochastic_depth')
    else:
        layer = layers.Activation('linear', name=name + '_identity')

    return layers.Add()([inputs, layer(x)])

def ConvNext(depths,  # tiny:[3,3,9,3]
             projection_dims,  # tiny:[96, 192, 384, 768],
             drop_path_rate=0.0,  # 随机深度概率，如果为0.0，图层缩放不会被使用
             layer_scale_init_value=1e-6,  # 缩放比例
             default_size=224,  # 默认输入图像大小
             model_name='convnext',  # 模型的可选名称
             include_preprocessing=True,  # 是否包含预处理
             include_top=True,  # 是否包含分类头
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,  # 分类个数
             classifier_activation='softmax'):  # 分类器激活
    img_input = layers.Input(shape=input_shape)

    inputs = img_input
    x = inputs

    # if include_preprocessing:
    #     x = PreStem(x, name=model_name)

    # Stem block:4*4,96,stride=4
    stem = tf.keras.Sequential(
        [
            layers.Conv2D(projection_dims[0],
                          kernel_size=(4, 4),
                          strides=4,
                          name=model_name + '_stem_conv'),
            layers.LayerNormalization(epsilon=1e-6, name=model_name + '_stem_layernorm')
        ],
        name=model_name + '_stem'
    )

    # Downsampling blocks
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = tf.keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6, name=model_name + '_downsampling_layernorm_' + str(i)),
                layers.Conv2D(projection_dims[i + 1],
                              kernel_size=(2, 2),
                              strides=2,
                              name=model_name + '_downsampling_conv_' + str(i))
            ],
            name=model_name + '_downsampling_block_' + str(i)
        )
        downsample_layers.append(downsample_layer)

    # Stochastic depth schedule.
    # This is referred from the original ConvNeXt codebase:
    # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))
    ]

    # First apply downsampling blocks and then apply ConvNeXt stages.
    cur = 0

    num_convnext_blocks = 4
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):  # depth:[3,3,9,3]
            x = ConvNextBlock(x,
                              projection_dim=projection_dims[i],
                              drop_path_rate=depth_drop_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value,
                              name=model_name + f"_stage_{i}_block_{j}")
        cur += depths[i]
    if include_top:
        x = layers.GlobalAveragePooling2D(name=model_name + '_head_gap')(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=model_name + '_head_layernorm')(x)
        x = layers.Dense(classes, name=model_name + '_head_dense')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    model = Model(inputs=inputs, outputs=x, name=model_name)
    # Load weights.
    # if weights == "imagenet":
    #     if include_top:
    #         file_suffix = ".h5"
    #         file_hash = WEIGHTS_HASHES[model_name][0]
    #     else:
    #         file_suffix = "_notop.h5"
    #         file_hash = WEIGHTS_HASHES[model_name][1]
    #     file_name = model_name + file_suffix
    #     weights_path = utils.data_utils.get_file(
    #         file_name,
    #         BASE_WEIGHTS_PATH + file_name,
    #         cache_subdir="models",
    #         file_hash=file_hash,
    #     )
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model

def ConvNeXtTiny(model_name='convnext-tiny',
                 include_top=True,
                 include_processing=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax'):
    return ConvNext(depths=MODEL_CONFIGS['tiny']['depths'],
                    projection_dims=MODEL_CONFIGS['tiny']['projection_dims'],
                    drop_path_rate=0.0,
                    layer_scale_init_value=1e-6,
                    default_size=MODEL_CONFIGS["tiny"]['default_size'],
                    model_name=model_name,
                    include_top=include_top,
                    include_preprocessing=include_processing,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    pooling=pooling,
                    classes=classes,
                    classifier_activation=classifier_activation
                    )

def ConvNeXtLarge(model_name='convnext-Large',
                 include_top=True,
                 include_processing=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=3,
                 classifier_activation='softmax'):
    
    return ConvNext(depths=MODEL_CONFIGS['large']['depths'],
                    projection_dims=MODEL_CONFIGS['large']['projection_dims'],
                    drop_path_rate=0.0,
                    layer_scale_init_value=1e-6,
                    default_size=MODEL_CONFIGS["large"]['default_size'],
                    model_name=model_name,
                    include_top=include_top,
                    include_preprocessing=include_processing,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    pooling=pooling,
                    classes=classes,
                    classifier_activation=classifier_activation
                    )

def ConvNeXtBase(model_name='convnext-Base',
                 include_top=True,
                 include_processing=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=3,
                 classifier_activation='softmax'):
    
    return ConvNext(depths=MODEL_CONFIGS['base']['depths'],
                    projection_dims=MODEL_CONFIGS['base']['projection_dims'],
                    drop_path_rate=0.0,
                    layer_scale_init_value=1e-6,
                    default_size=MODEL_CONFIGS["base"]['default_size'],
                    model_name=model_name,
                    include_top=include_top,
                    include_preprocessing=include_processing,
                    weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    pooling=pooling,
                    classes=classes,
                    classifier_activation=classifier_activation
                    )

if __name__ == '__main__':
    model = ConvNeXtLarge(input_shape=(128, 128, 3))
    model.summary()