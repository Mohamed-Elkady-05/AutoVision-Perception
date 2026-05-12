"""
VGG16 Feature Extraction and Caching
Extracts CNN features from video sequences and caches them for efficient reuse.
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

from src.config import FeatureExtractorConfig, DatasetConfig


class VGG16FeatureExtractor:
    """
    Extracts features from VGG16 pretrained model.
    
    Design:
    - Load pretrained VGG16 (ImageNet)
    - Extract features from intermediate layer (layer_30 → 512 dims)
    - Cache extracted features for efficient batch processing
    - Support both single images and image sequences
    
    Usage:
        extractor = VGG16FeatureExtractor()
        # Single image
        image = Image.open("sign.jpg")
        feature = extractor.extract_single(image)  # shape: (512,)
        
        # Sequence of images
        images = [Image.open(f"frame_{i}.jpg") for i in range(10)]
        features = extractor.extract_sequence(images)  # shape: (10, 512)
        
        # Batch with caching
        features = extractor.extract_and_cache(image_paths, cache_key)
    """
    
    def __init__(self, config: FeatureExtractorConfig = None, device: str = None):
        """
        Initialize VGG16 feature extractor.
        
        Args:
            config: FeatureExtractorConfig instance (optional)
            device: "cuda" or "cpu" (overrides config if provided)
        """
        self.config = config or FeatureExtractorConfig()
        self.device = device or self.config.DEVICE
        
        # Load pretrained VGG16
        print(f"[FeatureExtractor] Loading VGG16 from torchvision...")
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.to(self.device)
        self.vgg16.eval()  # evaluation mode
        
        # Extract feature extractor part (remove classifier)
        # VGG16 structure: features (conv layers) + avgpool + classifier (fc layers)
        # We use only features (conv layers) and apply adaptive pooling separately
        self.feature_extractor = self.vgg16.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Preprocessing transforms (ImageNet normalization)
        self.transforms = transforms.Compose([
            transforms.Resize(DatasetConfig.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DatasetConfig.NORMALIZE_MEAN,
                std=DatasetConfig.NORMALIZE_STD
            )
        ])
        
        print(f"[FeatureExtractor] Initialized on device: {self.device}")
        print(f"[FeatureExtractor] Feature dimension: {self.config.FEATURE_DIM}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess single PIL Image to tensor.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transforms(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    @torch.no_grad()
    def extract_single(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Feature vector of shape (512,) as numpy array
        """
        tensor = self._preprocess_image(image).to(self.device)  # (1, 3, 224, 224)
        features = self.feature_extractor(tensor)  # (1, 512, 7, 7)
        features = self.pool(features)  # (1, 512, 1, 1)
        features = self.flatten(features)  # (1, 512)
        features = features.squeeze(0).cpu().numpy()  # (512,)
        
        return features
    
    @torch.no_grad()
    def extract_sequence(self, images: List) -> np.ndarray:
        """
        Extract features from sequence of images.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            Feature array of shape (seq_len, 512) as numpy array
        """
        features_list = []
        
        for img in images:
            tensor = self._preprocess_image(img).to(self.device)  # (1, 3, 224, 224)
            features = self.feature_extractor(tensor)  # (1, 512, 7, 7)
            features = self.pool(features)  # (1, 512, 1, 1)
            features = self.flatten(features)  # (1, 512)
            features = features.squeeze(0).cpu().numpy()  # (512,)
            features_list.append(features)
        
        return np.array(features_list)  # shape: (seq_len, 512)
    
    @torch.no_grad()
    def extract_batch(self, image_paths: List[str], batch_size: int = None) -> np.ndarray:
        """
        Extract features from batch of image files with progress.
        
        Args:
            image_paths: List of file paths to images
            batch_size: Batch size for processing (default: config.BATCH_SIZE)
            
        Returns:
            Feature array of shape (num_images, 512)
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [Image.open(p).convert('RGB') for p in batch_paths]
            
            # Stack into batch
            tensors = [self.transforms(img).unsqueeze(0) for img in batch_images]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)  # (batch_size, 3, 224, 224)
            
            # Extract features
            features = self.feature_extractor(batch_tensor)  # (batch_size, 512, 7, 7)
            features = self.pool(features)  # (batch_size, 512, 1, 1)
            features = self.flatten(features)  # (batch_size, 512)
            features = features.cpu().numpy()
            
            all_features.append(features)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"[FeatureExtractor] Processed {i + len(batch_paths)} / {len(image_paths)} images")
        
        return np.vstack(all_features)  # shape: (num_images, 512)
    
    # ========================================================================
    # CACHING UTILITIES
    # ========================================================================
    
    @staticmethod
    def _cache_path(cache_dir: Path, cache_key: str) -> Path:
        """Generate cache file path."""
        return cache_dir / f"{cache_key}_features.pkl"
    
    def load_cache(self, cache_key: str, cache_dir: Path = None) -> Optional[np.ndarray]:
        """
        Load features from cache.
        
        Args:
            cache_key: Unique identifier for cached features
            cache_dir: Directory containing cache (default: config.CACHE_DIR)
            
        Returns:
            Feature array or None if not found
        """
        cache_dir = Path(cache_dir or self.config.CACHE_DIR)
        cache_file = self._cache_path(cache_dir, cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                print(f"[FeatureExtractor] Loaded cache: {cache_key}")
                return features
            except Exception as e:
                print(f"[FeatureExtractor] Error loading cache {cache_key}: {e}")
                return None
        
        return None
    
    def save_cache(self, features: np.ndarray, cache_key: str, cache_dir: Path = None):
        """
        Save features to cache.
        
        Args:
            features: Feature array to cache
            cache_key: Unique identifier for cached features
            cache_dir: Directory for cache (default: config.CACHE_DIR)
        """
        cache_dir = Path(cache_dir or self.config.CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self._cache_path(cache_dir, cache_key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            print(f"[FeatureExtractor] Saved cache: {cache_key}")
        except Exception as e:
            print(f"[FeatureExtractor] Error saving cache {cache_key}: {e}")
    
    def extract_and_cache(self, image_paths: List[str], cache_key: str, 
                         cache_dir: Path = None, force_recompute: bool = False) -> np.ndarray:
        """
        Extract features with caching. If cached, load from disk. Otherwise compute and save.
        
        Args:
            image_paths: List of image file paths
            cache_key: Unique identifier for this feature set
            cache_dir: Cache directory (default: config.CACHE_DIR)
            force_recompute: Skip cache and recompute if True
            
        Returns:
            Feature array of shape (num_images, 512)
        """
        cache_dir = Path(cache_dir or self.config.CACHE_DIR)
        
        # Try to load from cache
        if not force_recompute:
            cached_features = self.load_cache(cache_key, cache_dir)
            if cached_features is not None:
                return cached_features
        
        # Compute features
        print(f"[FeatureExtractor] Computing features for {cache_key}...")
        features = self.extract_batch(image_paths)
        
        # Cache if enabled
        if self.config.CACHE_FEATURES:
            self.save_cache(features, cache_key, cache_dir)
        
        return features


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def precompute_gtsrb_features(train_dir: str, val_dir: str, test_dir: str, 
                              cache_dir: Path = None, device: str = "cuda"):
    """
    Precompute and cache VGG16 features for entire GTSRB dataset.
    
    Args:
        train_dir: Path to training images
        val_dir: Path to validation images
        test_dir: Path to test images
        cache_dir: Cache directory
        device: Device for extraction
    """
    extractor = VGG16FeatureExtractor(device=device)
    cache_dir = Path(cache_dir or FeatureExtractorConfig.CACHE_DIR)
    
    splits = {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    }
    
    for split_name, split_dir in splits.items():
        if os.path.exists(split_dir):
            image_paths = [
                str(p)
                for pattern in ("*.ppm", "*.png", "*.jpg", "*.jpeg")
                for p in Path(split_dir).rglob(pattern)
            ]
            if image_paths:
                extractor.extract_and_cache(
                    image_paths,
                    cache_key=f"gtsrb_{split_name}",
                    cache_dir=cache_dir
                )


if __name__ == "__main__":
    # Test feature extraction
    from PIL import Image
    
    extractor = VGG16FeatureExtractor()
    
    # Create dummy image for testing
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    # Extract from single image
    features = extractor.extract_single(dummy_img)
    print(f"Single image features shape: {features.shape}")
    assert features.shape == (512,), f"Expected (512,), got {features.shape}"
    
    # Extract from sequence
    sequence = [dummy_img for _ in range(10)]
    seq_features = extractor.extract_sequence(sequence)
    print(f"Sequence features shape: {seq_features.shape}")
    assert seq_features.shape == (10, 512), f"Expected (10, 512), got {seq_features.shape}"
    
    print("\n✓ Feature extraction tests passed!")
