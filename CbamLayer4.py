import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Multiply, Add, Conv2D, Input, Activation, Concatenate
 

class CBAM4(Layer):
    def __init__(self, reduction_ratio=8, use_channel_att=True, use_spatial_att=True, **kwargs):
        super(CBAM4, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.use_channel_att = use_channel_att
        self.use_spatial_att = use_spatial_att

    def build(self, input_shape):
        channels = input_shape[-1]
        
        self.channel_attention_module_Dense1 = Dense(channels//self.reduction_ratio, activation="relu", use_bias=False)
        self.channel_attention_module_Dense2 = Dense(channels, use_bias=False)
        self.channel_attention_module_GlobalAveragePooling2D = GlobalAveragePooling2D()
        self.channel_attention_module_GlobalMaxPooling2D = GlobalMaxPooling2D()
        self.channel_attention_module_Activation = Activation("sigmoid")
        self.channel_attention_module_Conv2D1 = Conv2D(16, (3, 3), padding='same', activation = 'relu')
        self.channel_attention_module_Conv2D2 = Conv2D(16, (3, 3), padding='same', activation='relu')
        self.channel_attention_module_Conv2D3 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.channel_attention_module_Conv2D4 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.spatial_attention_module_Conv2D = Conv2D(1, kernel_size=(3,3), padding="same", activation="sigmoid")
          

         


    def call(self, inputs):
        #channel_attention_module
        ## Conv layer 
        z = self.channel_attention_module_Conv2D1(inputs)
        z= self.channel_attention_module_Conv2D2(z)
        z= self.channel_attention_module_Conv2D3(z)
        z = self.channel_attention_module_Conv2D4(z)

        ## Global Average Pooling
        x1 = self.channel_attention_module_GlobalAveragePooling2D(z)
        x1 = self.channel_attention_module_Dense1(x1)
        x1 = self.channel_attention_module_Dense2(x1)

        ## Global Max Pooling
        x2 = self.channel_attention_module_GlobalMaxPooling2D(z)
        x2 = self.channel_attention_module_Dense1(x2)
        x2 = self.channel_attention_module_Dense2(x2)

        ## Add both the features and pass through sigmoid
        feats = x1 + x2
        feats = self.channel_attention_module_Activation(feats)
        feats = Multiply()([z, feats])
               


        #spatial_attention_module
        ## Average Pooling
        x3 = tf.reduce_mean(feats, axis=-1)
        x3 = tf.expand_dims(x3, axis=-1)

        ## Max Pooling
        x4 = tf.reduce_max(feats, axis=-1)
        x4 = tf.expand_dims(x4, axis=-1)

        ## Concatenat both the features
        feats2 = Concatenate()([x3, x4]) 
        ## Conv layer 
        feats2 = self.spatial_attention_module_Conv2D(feats2)
        feats2 = Multiply()([feats, feats2])

        return feats2

     
    def get_config(self):
        config = super(CBAM4, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio,
                       'use_channel_att': self.use_channel_att,
                       'use_spatial_att': self.use_spatial_att})
        return config