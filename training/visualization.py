import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

def plot_training_history(history, save_path=None):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot IoU
    if 'iou' in history:
        axes[0, 1].plot(history['iou'], label='Training IoU')
        if 'val_iou' in history:
            axes[0, 1].plot(history['val_iou'], label='Validation IoU')
        axes[0, 1].set_title('IoU Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot Dice
    if 'dice' in history:
        axes[1, 0].plot(history['dice'], label='Training Dice')
        if 'val_dice' in history:
            axes[1, 0].plot(history['val_dice'], label='Validation Dice')
        axes[1, 0].set_title('Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot Accuracy
    if 'accuracy' in history:
        axes[1, 1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[1, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1, 1].set_title('Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def visualize_predictions(model, dataset, num_samples=4, save_path=None):
    """Visualize model predictions"""
    
    # Get samples from dataset
    samples = []
    for batch in dataset.take(1):
        images, masks = batch
        for i in range(min(num_samples, len(images))):
            samples.append((images[i], masks[i]))
        break
    
    # Make predictions
    predictions = []
    for image, mask in samples:
        pred = model.predict(tf.expand_dims(image, 0))[0]
        predictions.append(pred)
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, ((image, mask), pred) in enumerate(zip(samples, predictions)):
        # Denormalize image for display
        display_image = image.numpy()
        if display_image.min() < 0:  # If normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            display_image = display_image * std + mean
        display_image = np.clip(display_image, 0, 1)
        
        # Display original image
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Display ground truth mask
        axes[i, 1].imshow(mask.numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Display prediction
        pred_binary = (pred.squeeze() > 0.5).astype(np.float32)
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    
    plt.show()

def create_overlay_visualization(image, mask, prediction, alpha=0.5):
    """Create overlay visualization of prediction on original image"""
    
    # Ensure image is in [0, 1] range
    if image.max() > 1:
        image = image / 255.0
    
    # Create colored masks
    # Ground truth in green
    gt_overlay = np.zeros_like(image)
    gt_overlay[:, :, 1] = mask.squeeze()  # Green channel
    
    # Prediction in red
    pred_overlay = np.zeros_like(image)
    pred_overlay[:, :, 0] = (prediction.squeeze() > 0.5).astype(np.float32)  # Red channel
    
    # Create overlays
    gt_result = image * (1 - alpha) + gt_overlay * alpha
    pred_result = image * (1 - alpha) + pred_overlay * alpha
    
    return gt_result, pred_result

def plot_model_comparison(models, dataset, model_names, num_samples=2):
    """Compare predictions from multiple models"""
    
    # Get samples
    samples = []
    for batch in dataset.take(1):
        images, masks = batch
        for i in range(min(num_samples, len(images))):
            samples.append((images[i], masks[i]))
        break
    
    # Get predictions from all models
    all_predictions = []
    for model in models:
        model_preds = []
        for image, mask in samples:
            pred = model.predict(tf.expand_dims(image, 0))[0]
            model_preds.append(pred)
        all_predictions.append(model_preds)
    
    # Create visualization
    num_cols = 2 + len(models)  # Image, GT, + predictions
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, (image, mask) in enumerate(samples):
        # Denormalize image
        display_image = image.numpy()
        if display_image.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            display_image = display_image * std + mean
        display_image = np.clip(display_image, 0, 1)
        
        # Original image
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(mask.numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Model predictions
        for j, (model_preds, model_name) in enumerate(zip(all_predictions, model_names)):
            pred = model_preds[i]
            pred_binary = (pred.squeeze() > 0.5).astype(np.float32)
            axes[i, 2 + j].imshow(pred_binary, cmap='gray')
            axes[i, 2 + j].set_title(model_name)
            axes[i, 2 + j].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_sample_predictions(model, dataset, save_dir, num_samples=10):
    """Save sample predictions to files"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    sample_count = 0
    for batch in dataset:
        if sample_count >= num_samples:
            break
            
        images, masks = batch
        predictions = model.predict(images)
        
        for i in range(len(images)):
            if sample_count >= num_samples:
                break
                
            image = images[i]
            mask = masks[i]
            pred = predictions[i]
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Denormalize image
            display_image = image.numpy()
            if display_image.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                display_image = display_image * std + mean
            display_image = np.clip(display_image, 0, 1)
            
            axes[0].imshow(display_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask.numpy().squeeze(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            pred_binary = (pred.squeeze() > 0.5).astype(np.float32)
            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{sample_count:03d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"Saved {sample_count} sample predictions to {save_dir}") 