import tensorflow as tf
import numpy as np

def augment_data(image, mask, config):
    """Apply data augmentation to image and mask pair for 64x64 images"""
    config = config.augmentation
    
    # Convert to tf.float32 if needed
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    # Random horizontal flip
    if config.horizontal_flip.enable and tf.random.uniform([]) < config.horizontal_flip.probability:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Random vertical flip
    if config.vertical_flip.enable and tf.random.uniform([]) < config.vertical_flip.probability:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Random rotation (improved version)
    if config.rotation.enable and tf.random.uniform([]) < config.rotation.probability:
        # Limit rotation angle for small images to avoid losing important information
        angle = tf.random.uniform([], -config.rotation.limit, config.rotation.limit)
        
        # Use more precise rotation for small images
        if hasattr(config.rotation, 'use_precise') and config.rotation.use_precise:
            # For precise rotation, you might need tensorflow-addons
            # image = tfa.image.rotate(image, angle * np.pi / 180.0, interpolation='bilinear')
            # mask = tfa.image.rotate(mask, angle * np.pi / 180.0, interpolation='nearest')
            pass
        else:
            # Simple 90-degree rotations only
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
            mask = tf.image.rot90(mask, k=k)
    
    # Random zoom/scale (careful with 64x64)
    if hasattr(config, 'zoom') and config.zoom.enable and tf.random.uniform([]) < config.zoom.probability:
        scale_factor = tf.random.uniform([], 1.0 - config.zoom.limit, 1.0 + config.zoom.limit)
        
        # Calculate new size
        new_size = tf.cast(64.0 * scale_factor, tf.int32)
        new_size = tf.clip_by_value(new_size, 32, 96)  # Reasonable bounds
        
        # Resize
        image = tf.image.resize(image, [new_size, new_size], method='bilinear')
        mask = tf.image.resize(mask, [new_size, new_size], method='nearest')
        
        # Crop or pad back to 64x64
        if new_size > 64:
            # Random crop
            image = tf.image.random_crop(image, [64, 64, tf.shape(image)[-1]])
            mask = tf.image.random_crop(mask, [64, 64, tf.shape(mask)[-1]])
        else:
            # Pad to center
            padding = (64 - new_size) // 2
            image = tf.image.resize_with_crop_or_pad(image, 64, 64)
            mask = tf.image.resize_with_crop_or_pad(mask, 64, 64)
    
    # Brightness adjustment (only for image, reduced intensity for small images)
    if config.brightness.enable and tf.random.uniform([]) < config.brightness.probability:
        # Reduce brightness change for small images
        limit = min(config.brightness.limit, 0.2)
        delta = tf.random.uniform([], -limit, limit)
        image = tf.image.adjust_brightness(image, delta)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Contrast adjustment (only for image, reduced intensity)
    if config.contrast.enable and tf.random.uniform([]) < config.contrast.probability:
        # Reduce contrast change for small images
        limit = min(config.contrast.limit, 0.3)
        factor = tf.random.uniform([], 1.0 - limit, 1.0 + limit)
        image = tf.image.adjust_contrast(image, factor)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Saturation adjustment (only for image)
    if config.saturation.enable and tf.random.uniform([]) < config.saturation.probability:
        limit = min(config.saturation.limit, 0.3)
        factor = tf.random.uniform([], 1.0 - limit, 1.0 + limit)
        image = tf.image.adjust_saturation(image, factor)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Hue adjustment (only for image)
    if config.hue.enable and tf.random.uniform([]) < config.hue.probability:
        limit = min(config.hue.limit, 0.1)  # Reduce hue change
        delta = tf.random.uniform([], -limit, limit)
        image = tf.image.adjust_hue(image, delta)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Gaussian noise (only for image, reduced for small images)
    if config.noise.enable and tf.random.uniform([]) < config.noise.probability:
        # Much smaller noise for 64x64 images
        noise_std = min(config.noise.limit, 0.02)
        noise = tf.random.normal(tf.shape(image), stddev=noise_std)
        image = image + noise
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Optional: Random shift/translation
    if hasattr(config, 'shift') and config.shift.enable and tf.random.uniform([]) < config.shift.probability:
        # Small shifts for 64x64 images
        max_shift = min(config.shift.limit, 8)  # Max 8 pixels
        shift_x = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
        shift_y = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
        
        image = tf.roll(image, shift_x, axis=1)
        image = tf.roll(image, shift_y, axis=0)
        mask = tf.roll(mask, shift_x, axis=1)
        mask = tf.roll(mask, shift_y, axis=0)
    
    return image, mask

