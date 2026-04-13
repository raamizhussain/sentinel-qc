"""
PatchCore Anomaly Detector for unknown defects
Detects defects not seen during training using nearest neighbor approach
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
import cv2
from PIL import Image
import pickle


class PatchCoreDetector:
    """
    PatchCore-inspired anomaly detector using pre-trained features
    Detects: unknown anomalies, deviations from normal patterns
    """
    
    def __init__(self, backbone: str = "resnet50", device: str = None):
        """
        Initialize PatchCore detector
        
        Args:
            backbone: Feature extraction backbone (resnet50, resnet18, etc.)
            device: torch device ('cpu', 'cuda', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = backbone
        
        # Load pre-trained feature extractor
        import torchvision.models as models
        
        if backbone == "resnet50":
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == "resnet18":
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Use features from layer3 (before final layer) for good locality
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-2])
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Disable gradients
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Storage for normal features (training)
        self.normal_features = None
        self.feature_dim = 1024  # ResNet50 layer3 output channels
        
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract features from image using pre-trained backbone
        
        Args:
            image: Input image (H, W, 3) in [0, 255]
            
        Returns:
            Feature map (1, C, H', W')
        """
        # Preprocess
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Normalize (ImageNet stats)
        image_tensor = torch.nn.functional.normalize(
            image_tensor.unsqueeze(0),
            p=2, dim=1
        ).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.encoder(image_tensor)  # (1, C, H', W')
        
        return features
    
    def fit(self, image_paths: List[str], subsample_factor: float = 0.1):
        """
        Train on normal (good) images
        
        Args:
            image_paths: List of paths to normal training images
            subsample_factor: Fraction of patches to keep (for efficiency)
        """
        print(f"Fitting PatchCore on {len(image_paths)} normal images...")
        
        all_features = []
        
        for img_path in image_paths:
            try:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize to standard size
                h, w = 640, 640
                image = cv2.resize(image, (w, h))
                
                # Extract features
                features = self.extract_features(image)  # (1, C, H', W')
                features = features.squeeze(0).cpu().numpy()  # (C, H', W')
                
                # Reshape to (H', W', C)
                features = np.transpose(features, (1, 2, 0))
                
                # Subsample patches for memory efficiency
                h_patches, w_patches = features.shape[:2]
                step = max(1, int(1 / np.sqrt(subsample_factor)))
                
                for i in range(0, h_patches, step):
                    for j in range(0, w_patches, step):
                        patch = features[i, j, :]
                        all_features.append(patch)
            
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
        
        # Store as reference set for neighbor search
        self.normal_features = np.vstack(all_features)
        print(f"Stored {len(self.normal_features)} patch features")
        
    def detect(self, image: np.ndarray, threshold: float = 0.3) -> Tuple[float, np.ndarray]:
        """
        Detect anomalies using nearest neighbor distance
        
        Args:
            image: Input image (H, W, 3) in [0, 255]
            threshold: Anomaly score threshold (0-1)
            
        Returns:
            (anomaly_score, anomaly_map)
            - anomaly_score: 0-1, 0=normal, 1=anomaly
            - anomaly_map: Spatial heatmap of anomalies
        """
        if self.normal_features is None:
            raise RuntimeError("Must call fit() first")
        
        h, w = 640, 640
        image = cv2.resize(image, (w, h))
        
        # Extract features
        features = self.extract_features(image)  # (1, C, H', W')
        features = features.squeeze(0).cpu().numpy()  # (C, H', W')
        features = np.transpose(features, (1, 2, 0))
        
        h_patches, w_patches = features.shape[:2]
        
        # Compute nearest neighbor distances
        from scipy.spatial.distance import cdist
        
        anomaly_map = np.zeros((h_patches, w_patches))
        
        for i in range(h_patches):
            for j in range(w_patches):
                patch = features[i, j, :]
                
                # Find minimum distance to any normal patch
                distances = cdist([patch], self.normal_features, metric='euclidean')
                min_distance = np.min(distances)
                
                # Normalize distance to [0, 1]
                # Empirically, normal patches have distance < 5
                anomaly_score = min(min_distance / 5.0, 1.0)
                anomaly_map[i, j] = anomaly_score
        
        # Overall anomaly score
        overall_score = np.mean(anomaly_map)
        
        # Upsample heatmap to original image size
        anomaly_map_upsampled = cv2.resize(
            anomaly_map,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        return float(overall_score), anomaly_map_upsampled
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'normal_features': self.normal_features,
            'backbone': self.backbone,
            'feature_dim': self.feature_dim,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.normal_features = checkpoint['normal_features']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test PatchCore on bottle dataset
    mvtec_root = Path("../data/mvtec/bottle")
    
    detector = PatchCoreDetector(backbone="resnet50")
    
    # Fit on normal images
    train_good = list((mvtec_root / "train" / "good").glob("*.png"))[:50]
    
    print("Training PatchCore...")
    detector.fit([str(p) for p in train_good])
    
    # Test on various images
    test_good = (mvtec_root / "test" / "good" / "000.png")
    test_defect = (mvtec_root / "test" / "broken_large" / "000.png")
    
    if test_good.exists():
        image = cv2.imread(str(test_good))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        score, heatmap = detector.detect(image)
        print(f"Good image - Anomaly score: {score:.3f}")
    
    if test_defect.exists():
        image = cv2.imread(str(test_defect))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        score, heatmap = detector.detect(image)
        print(f"Defect image - Anomaly score: {score:.3f}")
    
    # Save model
    detector.save("../models/patchcore_bottle.pkl")
