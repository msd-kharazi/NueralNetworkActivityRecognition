import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Conv2D 

class CBAM(Layer):
    def __init__(self, reduction_ratio=8, use_channel_att=True, use_spatial_att=True, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.use_channel_att = use_channel_att
        self.use_spatial_att = use_spatial_att

    def build(self, input_shape):
        channels = input_shape[-1]
        if self.use_channel_att:
            self.channel_conv1 = Conv2D(channels // self.reduction_ratio, 1, activation='relu')
            self.channel_conv2 = Conv2D(channels, 1, activation='sigmoid')
        if self.use_spatial_att:
            self.spatial_conv1 = Conv2D(1, 1, activation='relu')
            self.spatial_conv2 = Conv2D(2, 1, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        if self.use_channel_att:
            x_channel = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            x_channel = self.channel_conv1(x_channel)
            x_channel = self.channel_conv2(x_channel)
            x_channel = Multiply()([x, x_channel])
        if self.use_spatial_att:
            x_spatial = self.spatial_conv1(x)
            x_spatial = self.spatial_conv2(x_spatial)
            x_spatial = Multiply()([x, x_spatial])
        x = Add()([x_channel, x_spatial])  # اضافه کردن همه‌ی توجه‌ها
        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio,
                       'use_channel_att': self.use_channel_att,
                       'use_spatial_att': self.use_spatial_att})
        return config