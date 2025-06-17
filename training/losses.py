import tensorflow as tf
from tensorflow import keras

class DiceLoss(keras.losses.Loss):
    """Dice Loss for segmentation (PyTorch-style)"""
    
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return dice loss
        return 1.0 - dice

class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance (PyTorch-style)"""
    
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent NaN
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Calculate cross entropy
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)

class CombinedLoss(keras.losses.Loss):
    """Combined BCE + Dice Loss (PyTorch-style)"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, **kwargs):
        super(CombinedLoss, self).__init__(**kwargs)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = keras.losses.BinaryCrossentropy()
        self.dice_loss = DiceLoss()
    
    def call(self, y_true, y_pred):
        bce = self.bce_loss(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        
        return self.bce_weight * bce + self.dice_weight * dice

class TverskyLoss(keras.losses.Loss):
    """Tversky Loss - generalization of Dice Loss (PyTorch-style)"""
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, **kwargs):
        super(TverskyLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # True positives, false positives, false negatives
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1.0 - tversky

class IoULoss(keras.losses.Loss):
    """Intersection over Union Loss (PyTorch-style)"""
    
    def __init__(self, smooth=1e-6, **kwargs):
        super(IoULoss, self).__init__(**kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        
        # IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou

# PyTorch-style factory functions
def dice_loss(smooth=1e-6):
    """Create Dice Loss"""
    return DiceLoss(smooth=smooth)

def focal_loss(alpha=1.0, gamma=2.0):
    """Create Focal Loss"""
    return FocalLoss(alpha=alpha, gamma=gamma)

def combined_loss(bce_weight=0.5, dice_weight=0.5):
    """Create Combined BCE + Dice Loss"""
    return CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight)

def tversky_loss(alpha=0.7, beta=0.3, smooth=1e-6):
    """Create Tversky Loss"""
    return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

def iou_loss(smooth=1e-6):
    """Create IoU Loss"""
    return IoULoss(smooth=smooth)

# Loss function registry (PyTorch-style)
LOSS_FUNCTIONS = {
    'dice': dice_loss,
    'focal': focal_loss,
    'combined': combined_loss,
    'tversky': tversky_loss,
    'iou': iou_loss,
    'bce': lambda: keras.losses.BinaryCrossentropy(),
    'mse': lambda: keras.losses.MeanSquaredError(),
}

def get_loss_function(name, **kwargs):
    """Get loss function by name (PyTorch-style)"""
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    
    return LOSS_FUNCTIONS[name](**kwargs) 