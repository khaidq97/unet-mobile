from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data.augmentation import augment_data

class NailDataset:
    def __init__(self, image_paths, mask_paths, config, transform=None, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        # Validate paths
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        
        self.length = len(image_paths)
    
    def __len__(self):
        return self.length
    
    def _load_sample(self, image_path, mask_path):
        """Load a single image-mask pair"""
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.uint8)  # Ensure uint8 dtype
        
        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.cast(mask, tf.uint8)  # Ensure uint8 dtype
        
        # Convert mask to binary
        mask = tf.where(mask > 0, 255, 0)
        mask = tf.cast(mask, tf.uint8)  # Ensure result is uint8
        
        return image, mask
    
    def _preprocess_sample(self, image, mask):
        """Preprocess image and mask"""
        # Resize to target size
        target_size = self.config.model.input_shape[:2]
        
        image = tf.image.resize(image, target_size, method='bilinear')
        mask = tf.image.resize(mask, target_size, method='nearest')
        
        # Normalize image
        image = tf.cast(image, tf.float32) / 255.0
        
        # Normalize mask to [0, 1]
        mask = tf.cast(mask, tf.float32) / 255.0
        
        # Apply dataset normalization if specified
        if self.config.data.normalize:
            mean = tf.constant(self.config.data.mean, dtype=tf.float32)
            std = tf.constant(self.config.data.std, dtype=tf.float32)
            image = (image - mean) / std
        
        return image, mask
    
    def get_tf_dataset(self):
        """Get TensorFlow dataset (PyTorch-style DataLoader equivalent)"""
        # Create dataset from paths
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        
        # Load images and masks using pure TensorFlow operations
        def load_and_preprocess(img_path, mask_path):
            # Load image
            image = tf.io.read_file(img_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.uint8)
            
            # Load mask
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
            mask = tf.cast(mask, tf.uint8)
            
            # Convert mask to binary
            mask = tf.where(mask > 0, 255, 0)
            mask = tf.cast(mask, tf.uint8)
            
            # Set shapes for resize operations
            image.set_shape([None, None, 3])
            mask.set_shape([None, None, 1])
            
            return image, mask
        
        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Preprocess first (resize, then ensure shape)
        dataset = dataset.map(
            self._preprocess_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes after preprocessing
        target_shape = self.config.model.input_shape
        dataset = dataset.map(lambda img, mask: (
            tf.ensure_shape(img, target_shape),
            tf.ensure_shape(mask, target_shape[:2] + [1,])
        ))
        
        # Apply augmentation if training
        if self.is_training and self.transform:
            def augment_wrapper(img, mask):
                # Apply augmentation (expects float32 inputs)
                aug_img, aug_mask = self.transform(img, mask, self.config)
                # Ensure shapes are maintained
                aug_img = tf.ensure_shape(aug_img, target_shape)
                aug_mask = tf.ensure_shape(aug_mask, target_shape[:2] + [1,])
                return aug_img, aug_mask
            
            dataset = dataset.map(
                augment_wrapper,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and shuffle
        if self.is_training and self.config.data.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.config.data.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
def load_image_mask_paths(config):
    image_paths = []
    mask_paths = []
    
    extensions = config.data.extensions
    for dataset_path in config.data.dataset_path:
        image_dir = Path(dataset_path) / config.data.image_dir
        mask_dir = Path(dataset_path) / config.data.mask_dir
        mask_imgs = list(mask_dir.rglob('*'))
        
        for img_path in image_dir.rglob('*'):
            if img_path.suffix.lower().replace('.', '') in extensions:
                for mask_img in mask_imgs:
                    if mask_img.stem == img_path.stem:
                        image_paths.append(str(img_path))
                        mask_paths.append(str(mask_img))
                        mask_imgs.pop(mask_imgs.index(mask_img))
                        break
    return image_paths, mask_paths
    

def create_datasets(config):
    # Get image and mask paths
    image_paths, mask_paths = load_image_mask_paths(config)
    print(f"Found {len(image_paths)} image-mask pairs")
    
    # Split data
    if config.data.test_split > 0:
        # Three-way split
        train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
            image_paths, mask_paths, 
            test_size=(config.data.val_split + config.data.test_split),
            random_state=config.seed
        )
        
        val_size = config.data.val_split / (config.data.val_split + config.data.test_split)
        val_imgs, test_imgs, val_masks, test_masks = train_test_split(
            temp_imgs, temp_masks,
            test_size=(1 - val_size),
            random_state=config.seed
        )
    else:
        # Two-way split
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_paths, mask_paths,
            test_size=config.data.val_split,
            random_state=config.seed
        )
        test_imgs = test_masks = []
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    # Create augmentation transform
    transform = None
    if config.augmentation.enable:
        transform = augment_data
    
    # Create datasets
    train_dataset = NailDataset(
        train_imgs, train_masks, config, 
        transform=transform, is_training=True
    )
    
    val_dataset = NailDataset(
        val_imgs, val_masks, config,
        transform=None, is_training=False
    )
    
    test_dataset = None
    if test_imgs:
        test_dataset = NailDataset(
            test_imgs, test_masks, config,
            transform=None, is_training=False
        )
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(config):
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Get TensorFlow datasets
    train_loader = train_dataset.get_tf_dataset()
    val_loader = val_dataset.get_tf_dataset()
    
    test_loader = None
    if test_dataset is not None:
        test_loader = test_dataset.get_tf_dataset()
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    from configs import Config
    config = Config()
    train_loader, val_loader, test_loader = create_data_loaders(config)
    for batch_images, batch_masks in train_loader.take(1):
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch masks shape: {batch_masks.shape}")
        break