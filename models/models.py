import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Register custom classes for serialization
@tf.keras.utils.register_keras_serializable()
class HardSwishMicro(layers.Layer):
    """Lightweight Hard Swish activation"""
    
    def __init__(self, **kwargs):
        super(HardSwishMicro, self).__init__(**kwargs)
    
    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0
    
    def get_config(self):
        """IMPORTANT: Fix serialization"""
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class SEBlockMicro(layers.Layer):
    """Ultra-lightweight Squeeze-and-Excitation block"""
    
    def __init__(self, channels, reduction=8, **kwargs):
        super(SEBlockMicro, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        
        # Use smaller reduction for tiny models
        reduced_channels = max(1, channels // reduction)
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(reduced_channels, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(channels, activation='sigmoid', use_bias=False)
        self.reshape = layers.Reshape((1, 1, channels))
    
    def call(self, x):
        se = self.global_pool(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return x * se
    
    def get_config(self):
        """IMPORTANT: Fix serialization"""
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class DepthwiseConvBlock(layers.Layer):
    """Efficient depthwise separable convolution block"""
    
    def __init__(self, out_channels, kernel_size=3, stride=1, use_se=False, **kwargs):
        super(DepthwiseConvBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size, 
            strides=stride, 
            padding='same', 
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.activation1 = HardSwishMicro()
        
        self.pointwise = layers.Conv2D(out_channels, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.activation2 = HardSwishMicro()
        
        if use_se:
            self.se = SEBlockMicro(out_channels)
    
    def call(self, x, training=None):
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        x = self.bn2(x, training=training)
        
        # SE block
        if self.use_se:
            x = self.se(x)
        
        x = self.activation2(x)
        return x
    
    def get_config(self):
        """IMPORTANT: Fix serialization"""
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'use_se': self.use_se,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class MicroEncoder(layers.Layer):
    """Ultra-lightweight encoder for small images"""
    
    def __init__(self, **kwargs):
        super(MicroEncoder, self).__init__(**kwargs)
        
        # Very shallow encoder for small images
        self.conv1 = layers.Conv2D(8, 3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.act1 = HardSwishMicro()
        
        # Downsampling blocks
        self.dwconv1 = DepthwiseConvBlock(16, stride=2)  
        self.dwconv2 = DepthwiseConvBlock(24, stride=2, use_se=True)
        self.dwconv3 = DepthwiseConvBlock(32, stride=2, use_se=True)
        self.dwconv4 = DepthwiseConvBlock(48, stride=2, use_se=True)
        
    def call(self, x, training=None):
        features = []
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        features.append(x)  # Full resolution
        
        # Downsample and collect features
        x = self.dwconv1(x, training=training)
        features.append(x)  # 1/2 resolution
        
        x = self.dwconv2(x, training=training)
        features.append(x)  # 1/4 resolution
        
        x = self.dwconv3(x, training=training)
        features.append(x)  # 1/8 resolution
        
        x = self.dwconv4(x, training=training)
        features.append(x)  # 1/16 resolution (bottleneck)
        
        return features
    
    def get_config(self):
        """IMPORTANT: Fix serialization"""
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class MicroUpBlock(layers.Layer):
    """Lightweight upsampling block"""
    
    def __init__(self, out_channels, **kwargs):
        super(MicroUpBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        
        self.up = layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv = DepthwiseConvBlock(out_channels)
        
    def call(self, x, skip=None, training=None):
        x = self.up(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                x = tf.image.resize(x, [skip.shape[1], skip.shape[2]], method='nearest')
            x = layers.concatenate([x, skip], axis=-1)
        
        x = self.conv(x, training=training)
        return x
    
    def get_config(self):
        """IMPORTANT: Fix serialization"""
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class UNetNano(keras.Model):
    """Nano version - even smaller for 64x64 images"""
    
    def __init__(self, num_classes=1, input_shape=(64, 64, 3), **kwargs):
        super(UNetNano, self).__init__(**kwargs)
        
        self.num_classes = num_classes
        self.model_input_shape = input_shape  # Rename to avoid conflict
        
        # Extremely simple encoder
        self.conv1 = layers.Conv2D(8, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(24, 3, strides=2, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        
        # Decoder
        self.up1 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv5 = layers.Conv2D(16, 3, padding='same', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        
        self.up2 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv6 = layers.Conv2D(8, 3, padding='same', use_bias=False)
        self.bn6 = layers.BatchNormalization()
        
        self.up3 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv7 = layers.Conv2D(8, 3, padding='same', use_bias=False)
        self.bn7 = layers.BatchNormalization()
        
        # Final layer
        self.final_conv = layers.Conv2D(
            num_classes, 
            1, 
            activation='sigmoid' if num_classes == 1 else 'softmax'
        )
        
    def call(self, x, training=None):
        # Encoder with skip connections
        x1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x2 = tf.nn.relu(self.bn2(self.conv2(x1), training=training))
        x3 = tf.nn.relu(self.bn3(self.conv3(x2), training=training))
        x4 = tf.nn.relu(self.bn4(self.conv4(x3), training=training))
        
        # Decoder with skip connections
        x = self.up1(x4)
        x = tf.concat([x, x3], axis=-1)
        x = tf.nn.relu(self.bn5(self.conv5(x), training=training))
        
        x = self.up2(x)
        x = tf.concat([x, x2], axis=-1)
        x = tf.nn.relu(self.bn6(self.conv6(x), training=training))
        
        x = self.up3(x)
        x = tf.concat([x, x1], axis=-1)
        x = tf.nn.relu(self.bn7(self.conv7(x), training=training))
        
        return self.final_conv(x)

    def get_config(self):
        """IMPORTANT: Fix serialization"""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.model_input_shape,  # Use renamed attribute
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        return cls(**config)

# ================================================================
# FIX 2: Đơn giản hóa UNetMicro - loại bỏ custom layers phức tạp
# ================================================================

@tf.keras.utils.register_keras_serializable()
class UNetMicro(keras.Model):
    """Ultra-lightweight U-Net optimized for small images (64x64 to 112x112)"""
    
    def __init__(self, num_classes=1, input_shape=(64, 64, 3), **kwargs):
        super(UNetMicro, self).__init__(**kwargs)
        
        self.num_classes = num_classes
        self.model_input_shape = input_shape  # Rename to avoid conflict
        
        # Simplified encoder - no custom layers
        self.conv1 = layers.Conv2D(8, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(24, 3, strides=2, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        
        self.conv5 = layers.Conv2D(48, 3, strides=2, padding='same', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        
        # Decoder
        self.up1 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv6 = layers.Conv2D(32, 3, padding='same', use_bias=False)
        self.bn6 = layers.BatchNormalization()
        
        self.up2 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv7 = layers.Conv2D(24, 3, padding='same', use_bias=False)
        self.bn7 = layers.BatchNormalization()
        
        self.up3 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv8 = layers.Conv2D(16, 3, padding='same', use_bias=False)
        self.bn8 = layers.BatchNormalization()
        
        self.up4 = layers.UpSampling2D(2, interpolation='nearest')
        self.conv9 = layers.Conv2D(8, 3, padding='same', use_bias=False)
        self.bn9 = layers.BatchNormalization()
        
        # Final classifier
        self.final_conv = layers.Conv2D(
            num_classes, 
            kernel_size=1, 
            activation='sigmoid' if num_classes == 1 else 'softmax',
            name='final_conv'
        )
    
    def call(self, x, training=None):
        # Encoder
        x1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))  # 64x64
        x2 = tf.nn.relu(self.bn2(self.conv2(x1), training=training))  # 32x32
        x3 = tf.nn.relu(self.bn3(self.conv3(x2), training=training))  # 16x16
        x4 = tf.nn.relu(self.bn4(self.conv4(x3), training=training))  # 8x8
        x5 = tf.nn.relu(self.bn5(self.conv5(x4), training=training))  # 4x4 (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5)  # 8x8
        x = tf.concat([x, x4], axis=-1)
        x = tf.nn.relu(self.bn6(self.conv6(x), training=training))
        
        x = self.up2(x)  # 16x16
        x = tf.concat([x, x3], axis=-1)
        x = tf.nn.relu(self.bn7(self.conv7(x), training=training))
        
        x = self.up3(x)  # 32x32
        x = tf.concat([x, x2], axis=-1)
        x = tf.nn.relu(self.bn8(self.conv8(x), training=training))
        
        x = self.up4(x)  # 64x64
        x = tf.concat([x, x1], axis=-1)
        x = tf.nn.relu(self.bn9(self.conv9(x), training=training))
        
        # Final classification
        x = self.final_conv(x)
        
        return x

    def get_config(self):
        """IMPORTANT: Fix serialization"""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.model_input_shape,  # Use renamed attribute
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        return cls(**config)

def unet_micro(num_classes=1, input_shape=(64, 64, 3), **kwargs):
    """Create Ultra-lightweight U-Net for small images"""
    return UNetMicro(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

def unet_nano(num_classes=1, input_shape=(64, 64, 3), **kwargs):
    """Create Nano U-Net for very small images and extreme efficiency"""
    return UNetNano(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )
