import os
import argparse
import logging
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Initialize logger
logger = logging.getLogger(__name__)

# Set environment variables to force CPU usage and reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide all GPUs from TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'

# Import custom modules (assuming same structure as training)
from configs import Config
from training.metrics import IoUMetric, DiceMetric

# Import model components safely
try:
    from models.models import (
        UNetMicro, UNetNano, HardSwishMicro, SEBlockMicro, 
        DepthwiseConvBlock, MicroEncoder, MicroUpBlock
    )
except ImportError as e:
    logger.warning(f"Could not import some model components: {e}")
    # Define empty classes as fallback
    UNetMicro = UNetNano = HardSwishMicro = SEBlockMicro = None
    DepthwiseConvBlock = MicroEncoder = MicroUpBlock = None

class ModelInference:
    """Class for handling model inference"""
    
    def __init__(self, model_path, config_path=None, device='cpu'):
        """
        Initialize inference class
        
        Args:
            model_path (str): Path to trained model
            config_path (str): Path to config file (optional)
            device (str): Device to use ('cpu' or 'gpu')
        """
        self.model_path = model_path
        self.config = Config(config_path) if config_path else None
        self.device = device
        self.model = None
        self.input_shape = None
        
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        try:
            # Configure TensorFlow based on device preference
            if self.device == 'cpu':
                tf.config.set_visible_devices([], 'GPU')
                
                # Set CPU threading configuration for better performance
                tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
                tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
                
                logger.info("TensorFlow configured to use CPU only")
            else:
                logger.info("TensorFlow will use available GPU if present")
            
            # Load model with custom objects
            custom_objects = {
                'IoUMetric': IoUMetric,
                'DiceMetric': DiceMetric,
            }
            
            # Add model components if they were imported successfully
            model_components = {
                'UNetMicro': UNetMicro,
                'UNetNano': UNetNano,
                'HardSwishMicro': HardSwishMicro,
                'SEBlockMicro': SEBlockMicro,
                'DepthwiseConvBlock': DepthwiseConvBlock,
                'MicroEncoder': MicroEncoder,
                'MicroUpBlock': MicroUpBlock
            }
            
            for name, component in model_components.items():
                if component is not None:
                    custom_objects[name] = component
            
            self.model = keras.models.load_model(
                self.model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            
            # Get input shape from model
            if hasattr(self.model, 'input_shape') and self.model.input_shape is not None:
                if len(self.model.input_shape) > 3:
                    self.input_shape = self.model.input_shape[1:3]  # (height, width)
                else:
                    # Fallback to default shape
                    self.input_shape = (64, 64)
                    logger.warning("Could not determine input shape from model, using default (64, 64)")
            else:
                # Default input shape
                self.input_shape = (64, 64)
                logger.warning("Model does not have input_shape attribute, using default (64, 64)")
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model input shape: {getattr(self.model, 'input_shape', 'Unknown')}")
            logger.info(f"Model output shape: {getattr(self.model, 'output_shape', 'Unknown')}")
            logger.info(f"Using input shape for preprocessing: {self.input_shape}")
            
            # Display device information
            self.display_device_info()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def display_device_info(self):
        """Display information about available devices"""
        try:
            # Get list of physical devices
            physical_devices = tf.config.list_physical_devices()
            logger.info(f"Available physical devices: {len(physical_devices)}")
            for device in physical_devices:
                logger.info(f"  - {device}")
            
            # Get list of logical devices
            logical_devices = tf.config.list_logical_devices()
            logger.info(f"Available logical devices: {len(logical_devices)}")
            for device in logical_devices:
                logger.info(f"  - {device}")
                
            # Check if running on CPU
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) == 0:
                logger.info("✓ Running inference on CPU")
            else:
                logger.info(f"Found {len(gpus)} GPU(s), but configured to use CPU only")
                
        except Exception as e:
            logger.warning(f"Could not display device info: {e}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            np.ndarray: Preprocessed image
            tuple: Original image shape
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path  # Assume it's already a numpy array
            
            original_shape = image.shape[:2]
            
            # Resize to model input size
            resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)  # (width, height)
            # Normalize to [0, 1]
            normalized_image = resized_image / 255.0
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized_image = (normalized_image - mean) / std
            
            # Add batch dimension
            batch_image = np.expand_dims(normalized_image, axis=0).astype(np.float32)
            return batch_image, original_shape
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def postprocess_mask(self, prediction, original_shape, threshold=0.5):
        """
        Postprocess prediction mask
        
        Args:
            prediction (np.ndarray): Model prediction
            original_shape (tuple): Original image shape
            threshold (float): Threshold for binary mask
            
        Returns:
            np.ndarray: Postprocessed mask
        """
        try:
            # Remove batch dimension
            mask = prediction[0]
            
            # Apply threshold if binary segmentation
            if mask.shape[-1] == 1:
                mask = (mask > threshold).astype(np.uint8)
                mask = mask.squeeze()
            else:
                # Multi-class segmentation
                mask = np.argmax(mask, axis=-1).astype(np.uint8)
            
            # Resize back to original shape
            if original_shape != mask.shape:
                mask = cv2.resize(mask, (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error postprocessing mask: {str(e)}")
            raise
    
    def predict_single_image(self, image_path, threshold=0.5):
        """
        Predict mask for a single image
        
        Args:
            image_path (str): Path to input image
            threshold (float): Threshold for binary mask
            
        Returns:
            dict: Prediction results
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            preprocessed_image, original_shape = self.preprocess_image(image_path)
            
            # Run inference
            prediction = self.model.predict(preprocessed_image, verbose=0)
            
            # Postprocess mask
            mask = self.postprocess_mask(prediction, original_shape, threshold)
            
            inference_time = time.time() - start_time
            
            # Calculate confidence score (mean probability)
            confidence = float(np.mean(prediction))
            
            results = {
                'mask': mask,
                'confidence': confidence,
                'inference_time': inference_time,
                'original_shape': original_shape,
                'prediction_shape': prediction.shape
            }
            
            logger.info(f"Inference completed in {inference_time:.3f}s")
            logger.info(f"Confidence score: {confidence:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in single image prediction: {str(e)}")
            raise
    
    def predict_batch(self, image_paths, threshold=0.5, batch_size=8):
        """
        Predict masks for multiple images
        
        Args:
            image_paths (list): List of image paths
            threshold (float): Threshold for binary mask
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
            
            for image_path in batch_paths:
                try:
                    result = self.predict_single_image(image_path, threshold)
                    result['image_path'] = image_path
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    continue
        
        return results
    
    def visualize_prediction(self, image_path, mask, save_path=None, show_plot=True):
        """
        Visualize prediction results
        
        Args:
            image_path (str): Path to original image
            mask (np.ndarray): Predicted mask
            save_path (str): Path to save visualization
            show_plot (bool): Whether to show plot
        """
        try:
            # Load original image
            original_image = cv2.imread(image_path)
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = original_image.copy()
            mask_colored = np.zeros_like(original_image)
            mask_colored[mask > 0] = [255, 0, 0]  # Red color for mask
            
            # Blend images
            overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Predicted Mask')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def save_results(results, output_dir):
    """Save inference results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save masks
    for i, result in enumerate(results):
        mask = result['mask']
        image_path = result.get('image_path', f'image_{i}')
        image_name = Path(image_path).stem
        
        # Save mask as PNG
        mask_path = os.path.join(output_dir, f'{image_name}_mask.png')
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        # Save mask as numpy array
        # npy_path = os.path.join(output_dir, f'{image_name}_mask.npy')
        # np.save(npy_path, mask)
    
    # Save summary
    summary = {
        'total_images': len(results),
        'average_confidence': np.mean([r['confidence'] for r in results]),
        'average_inference_time': np.mean([r['inference_time'] for r in results]),
        'results': [
            {
                'image_path': r.get('image_path', ''),
                'confidence': r['confidence'],
                'inference_time': r['inference_time']
            }
            for r in results
        ]
    }
    
    summary_path = os.path.join(output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Run inference on nail segmentation model')
    parser.add_argument('--model-path', type=str, default='./assets/models/best_model.keras',
                       help='Path to trained model')
    parser.add_argument('--input', type=str, default='./dataset/sample/images',
                       help='Input image path or directory')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing multiple images')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'gpu'],
                       help='Device to use for inference (cpu or gpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Configure device before importing tensorflow modules
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("Forcing CPU usage via CUDA_VISIBLE_DEVICES")
    elif args.device == 'gpu':
        # Remove CPU-only constraint if GPU is requested
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
            del os.environ['CUDA_VISIBLE_DEVICES']
        logger.info("GPU usage enabled")
    
    # Initialize inference
    logger.info("Initializing model inference...")
    inference = ModelInference(args.model_path, args.config_file)
    
    # Prepare input paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        # Get all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        image_paths = [str(p) for p in image_paths]
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Run inference
    logger.info("Starting inference...")
    results = inference.predict_batch(
        image_paths, 
        threshold=args.threshold, 
        batch_size=args.batch_size
    )
    
    # Save results
    save_results(results, args.output_dir)
    
    # Generate visualizations
    if args.visualize or args.save_viz:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        for result in results:
            if 'image_path' in result:
                image_name = Path(result['image_path']).stem
                viz_path = os.path.join(viz_dir, f'{image_name}_viz.png') if args.save_viz else None
                
                inference.visualize_prediction(
                    result['image_path'],
                    result['mask'],
                    save_path=viz_path,
                    show_plot=args.visualize
                )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total images processed: {len(results)}")
    logger.info(f"Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    logger.info(f"Average inference time: {np.mean([r['inference_time'] for r in results]):.3f}s")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Inference completed!")

if __name__ == '__main__':
    main()