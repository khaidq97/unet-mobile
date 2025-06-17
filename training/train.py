import os
import argparse
import logging

# Initialize logger (will be configured later)
logger = logging.getLogger(__name__)

# Set environment variables to reduce CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all but FATAL messages
os.environ['CUDA_CACHE_DISABLE'] = '1'    # Disable CUDA cache to reduce register usage
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'  # Completely disable XLA
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    # Force GPU memory growth
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'  # Disable NVTX profiling

# Configure GPU BEFORE importing TensorFlow
import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.random.set_seed(42)
np.random.seed(42)

from configs import Config
from models import get_model
from data.dataset import create_data_loaders
from training.losses import get_loss_function
from training.metrics import IoUMetric, DiceMetric
from training.visualization import plot_training_history

class TrainingLogger(keras.callbacks.Callback):
    """Custom callback for detailed training logging"""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
    
    def on_train_begin(self, logs=None):
        logger.info("="*50)
        logger.info("TRAINING STARTED")
        logger.info("="*50)
    
    def on_train_end(self, logs=None):
        logger.info("="*50)
        logger.info("TRAINING COMPLETED")
        logger.info("="*50)
    
    def on_epoch_begin(self, epoch, logs=None):
        import time
        self.epoch_start_time = time.time()
        logger.info(f"\n--- EPOCH {epoch + 1} STARTED ---")
    
    def on_epoch_end(self, epoch, logs=None):
        import time
        epoch_time = time.time() - self.epoch_start_time
        
        logger.info(f"--- EPOCH {epoch + 1} COMPLETED ---")
        logger.info(f"Epoch duration: {epoch_time:.2f}s")
        
        if logs:
            # Log training metrics
            logger.info("Training metrics:")
            for key, value in logs.items():
                if not key.startswith('val_'):
                    logger.info(f"  {key}: {value:.4f}")
            
            # Log validation metrics
            val_metrics = {k: v for k, v in logs.items() if k.startswith('val_')}
            if val_metrics:
                logger.info("Validation metrics:")
                for key, value in val_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")
        
        logger.info("-" * 30)
    
    def on_batch_begin(self, batch, logs=None):
        if batch % 50 == 0:  # Log every 50 batches
            logger.info(f"Processing batch {batch}...")
    
    def on_batch_end(self, batch, logs=None):
        if batch % 50 == 0 and logs:  # Log every 50 batches
            logger.info(f"Batch {batch} - Loss: {logs.get('loss', 0):.4f}")

class LearningRateLogger(keras.callbacks.Callback):
    """Custom callback for logging learning rate changes"""
    
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        logger.info(f"Learning rate for epoch {epoch + 1}: {lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Check if learning rate changed
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        if hasattr(self, 'previous_lr') and lr != self.previous_lr:
            logger.info(f"Learning rate changed: {self.previous_lr:.6f} -> {lr:.6f}")
        self.previous_lr = lr

class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_training()
    
    def setup_training(self):
        """Setup training environment"""
        # Set random seed (update if different from default)
        if hasattr(self.config, 'seed') and self.config.seed != 42:
            tf.random.set_seed(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Updated random seed to {self.config.seed}")
        
        # Setup mixed precision
        if self.config.training.mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            logger.info("Using mixed precision training")
        
        # Create model
        self.model = get_model(
                model_name=self.config.model.name,
                num_classes=self.config.model.num_classes,
                input_shape=self.config.model.input_shape,
                pretrained=self.config.model.pretrained
            )
        
        # Setup optimizer
        optimizer = self.config.training.optimizer.lower()
        if optimizer == 'adam':
            self.optimizer = keras.optimizers.Adam(
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer == 'sgd':
            self.optimizer = keras.optimizers.SGD(
                learning_rate=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Setup loss function
        self.loss_fn = get_loss_function(self.config.training.loss_function)
        
        # Setup metrics
        self.metrics = [
            IoUMetric(name='iou'),
            DiceMetric(name='dice'),
            keras.metrics.BinaryAccuracy(name='accuracy')
        ]
        
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics
        )
        # Build model
        input_shape = self.config.model.input_shape
        input_tensor = keras.Input(shape=input_shape)
        self.model(input_tensor)
        
        # Print model summary
        logger.info("\nModel Summary:")
        self.model.summary()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        self.callbacks = []
        
        # Add custom training logger
        training_logger = TrainingLogger()
        self.callbacks.append(training_logger)
        
        # Add learning rate logger
        lr_logger = LearningRateLogger()
        self.callbacks.append(lr_logger)
        
        # Model checkpoint
        if self.config.training.model_checkpoint.enable:
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.training.log_dir, 'best_model.keras'),
                monitor=self.config.training.model_checkpoint.monitor,
                mode=self.config.training.model_checkpoint.mode,
                save_best_only=self.config.training.model_checkpoint.save_best_only,
                save_weights_only=self.config.training.model_checkpoint.save_weights_only,
                verbose=self.config.training.model_checkpoint.verbose
            )
            self.callbacks.append(checkpoint_callback)
            logger.info(f"Model checkpoint enabled - monitoring: {self.config.training.model_checkpoint.monitor}")
        
        # Early stopping
        if self.config.training.early_stopping.enable:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=self.config.training.early_stopping.monitor,
                mode=self.config.training.early_stopping.mode,
                patience=self.config.training.early_stopping.patience,
                restore_best_weights=self.config.training.early_stopping.restore_best_weights,
                verbose=self.config.training.early_stopping.verbose
            )
            self.callbacks.append(early_stopping)
            logger.info(f"Early stopping enabled - patience: {self.config.training.early_stopping.patience}")
        
        # Learning rate reduction
        if self.config.training.learning_rate_reduction.enable:
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=self.config.training.learning_rate_reduction.factor,
                patience=self.config.training.learning_rate_reduction.patience,
                min_lr=self.config.training.learning_rate_reduction.min_lr,
                verbose=1
            )
            self.callbacks.append(lr_scheduler)
            logger.info(f"Learning rate reduction enabled - factor: {self.config.training.learning_rate_reduction.factor}")
        
        # TensorBoard logging
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.config.training.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        self.callbacks.append(tensorboard_callback)
        logger.info(f"TensorBoard logging enabled - log directory: {self.config.training.log_dir}")
    
    def train(self, train_loader, val_loader):
        """Train the model (PyTorch-style)"""
        logger.info(f"\nStarting training for {self.config.training.epochs} epochs...")
        
        # Log dataset information
        logger.info("Dataset Information:")
        logger.info(f"  Training batches: {len(train_loader) if train_loader else 0}")
        logger.info(f"  Validation batches: {len(val_loader) if val_loader else 0}")
        
        # Log optimizer information
        logger.info("Optimizer Configuration:")
        logger.info(f"  Type: {type(self.optimizer).__name__}")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        if hasattr(self.config.training, 'weight_decay'):
            logger.info(f"  Weight decay: {self.config.training.weight_decay}")
        
        # Train model
        history = self.model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=self.config.training.epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        if test_loader is None:
            logger.info("No test data available")
            return None
        
        logger.info("\nEvaluating on test set...")
        test_results = self.model.evaluate(test_loader, verbose=1)
        
        # Print results
        for i, metric_name in enumerate(['loss'] + [m.name for m in self.metrics]):
            logger.info(f"Test {metric_name}: {test_results[i]:.4f}")
        
        return test_results
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

def setup_logging(log_dir):
    """Setup logging configuration"""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(log_dir, 'log.log')
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ],
        force=True  # Force reconfiguration
    )
    
    logger.info(f"Logging configured - log file: {log_file}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train nail segmentation model')
    parser.add_argument('--config-file', type=str, default='configs/config.yml', 
                       help='Configuration file to use')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Log directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config_file:
        config = Config(args.config_file)
    else:
        config = Config()
        
    if args.log_dir:
        config.training.log_dir = args.log_dir
    
    # Setup logging using config
    setup_logging(config.training.log_dir)
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model.name}")
    logger.info(f"  Input shape: {config.model.input_shape}")
    logger.info(f"  Dataset: {config.data.dataset_path}")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Epochs: {config.training.epochs}")
    
    # Create data loaders
    logger.info("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Log training summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    
    # Get final metrics
    final_metrics = {}
    if history.history:
        for key, values in history.history.items():
            if values:
                final_metrics[key] = values[-1]
        
        logger.info("Final Training Metrics:")
        for key, value in final_metrics.items():
            if not key.startswith('val_'):
                logger.info(f"  {key}: {value:.4f}")
        
        val_metrics = {k: v for k, v in final_metrics.items() if k.startswith('val_')}
        if val_metrics:
            logger.info("Final Validation Metrics:")
            for key, value in val_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
    
    # Evaluate on test set
    if test_loader is not None:
        trainer.evaluate(test_loader)
    
    log_dir = config.training.log_dir
    # Save final model
    trainer.save_model(os.path.join(log_dir, 'final_model.keras'))
    
    # Plot training history
    plot_training_history(history.history, save_path=os.path.join(log_dir, 'training_history.png'))
    logger.info(f"Training history plot saved to: {os.path.join(log_dir, 'training_history.png')}")
    
    logger.info("\nTraining completed!")

if __name__ == '__main__':
    main() 