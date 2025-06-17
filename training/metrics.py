import tensorflow as tf
from tensorflow import keras

class IoUMetric(keras.metrics.Metric):
    """Intersection over Union metric for segmentation"""
    
    def __init__(self, threshold=0.5, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary
        y_pred_binary = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred_binary, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
        
        # Avoid division by zero
        iou = tf.where(
            tf.equal(union, 0),
            1.0,  # Perfect score when both are empty
            intersection / union
        )
        
        self.iou_sum.assign_add(iou)
        self.count.assign_add(1)
    
    def result(self):
        return self.iou_sum / self.count
    
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)

class DiceMetric(keras.metrics.Metric):
    """Dice coefficient metric for segmentation"""
    
    def __init__(self, threshold=0.5, smooth=1e-6, name='dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary
        y_pred_binary = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred_binary, [-1])
        
        # Calculate Dice coefficient
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1)
    
    def result(self):
        return self.dice_sum / self.count
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

class PixelAccuracyMetric(keras.metrics.Metric):
    """Pixel-wise accuracy metric"""
    
    def __init__(self, threshold=0.5, name='pixel_accuracy', **kwargs):
        super(PixelAccuracyMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct_sum = self.add_weight(name='correct_sum', initializer='zeros')
        self.total_sum = self.add_weight(name='total_sum', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary
        y_pred_binary = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate correct predictions
        correct = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
        
        self.correct_sum.assign_add(tf.reduce_sum(correct))
        self.total_sum.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.correct_sum / self.total_sum
    
    def reset_state(self):
        self.correct_sum.assign(0.0)
        self.total_sum.assign(0.0)

# Standalone metric functions (PyTorch-style)
def calculate_iou(y_true, y_pred, threshold=0.5):
    """Calculate IoU score"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Flatten tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred_binary, [-1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
    
    # Avoid division by zero
    iou = tf.where(
        tf.equal(union, 0),
        1.0,  # Perfect score when both are empty
        intersection / union
    )
    
    return iou

def calculate_dice(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Flatten tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred_binary, [-1])
    
    # Calculate Dice coefficient
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice

def calculate_pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Calculate pixel-wise accuracy"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate correct predictions
    correct = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
    accuracy = tf.reduce_mean(correct)
    
    return accuracy 