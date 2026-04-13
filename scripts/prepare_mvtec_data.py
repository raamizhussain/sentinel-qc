"""Prepare MVTec AD dataset for YOLOv10 training."""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image


class MVTecDatasetPreparator:
    """Convert MVTec AD anomaly detection format to YOLO detection format."""
    
    def __init__(self, mvtec_root: str = r"C:\Users\User\sentinel\data\mvtec"):
        """
        Initialize preparator.
        
        Args:
            mvtec_root: Path to MVTec AD root directory
        """
        self.mvtec_root = Path(mvtec_root)
        self.categories = [d.name for d in self.mvtec_root.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
    
    def get_ground_truth_boxes(self, mask_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes from ground truth mask.
        
        Args:
            mask_path: Path to ground truth mask image
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return []
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 5 and h > 5:  # Filter tiny boxes
                    boxes.append((x, y, w, h))
            
            return boxes
        except Exception as e:
            print(f"Error extracting boxes from {mask_path}: {e}")
            return []
    
    def mask_to_yolo_format(self, image_path: str, mask_path: str) -> str:
        """
        Convert mask to YOLO format (normalized class, x_center, y_center, width, height).
        Returns YOLO format string for one image.
        
        Args:
            image_path: Path to image
            mask_path: Path to ground truth mask
            
        Returns:
            YOLO format annotation string
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            img_h, img_w = image.shape[:2]
            
            boxes = self.get_ground_truth_boxes(mask_path)
            if not boxes:
                return ""
            
            yolo_lines = []
            for x, y, w, h in boxes:
                # Normalize to [0, 1]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h
                
                # Clamp values
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                # Class 0 = defect, Class 1 = good
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            return "\n".join(yolo_lines) if yolo_lines else ""
        
        except Exception as e:
            print(f"Error converting {image_path}: {e}")
            return ""
    
    def prepare_category(self, category: str, output_root: str = r"C:\Users\User\sentinel\data\yolo_mvtec"):
        """
        Prepare a single MVTec category for YOLO training.
        
        Args:
            category: Category name (e.g., 'bottle')
            output_root: Root directory for YOLO format dataset
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        
        category_dir = self.mvtec_root / category
        if not category_dir.exists():
            print(f"Category {category} not found")
            return
        
        # Create output structure
        train_images = output_root / "images" / "train"
        train_labels = output_root / "labels" / "train"
        val_images = output_root / "images" / "val"
        val_labels = output_root / "labels" / "val"
        
        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Process training set (only good images)
        train_dir = category_dir / "train" / "good"
        if train_dir.exists():
            for img_file in sorted(train_dir.glob("*.png")):
                # Copy good images to train set (class 1)
                dest_img = train_images / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Create label file with class 1 (good)
                label_file = train_labels / img_file.stem
                with open(label_file.with_suffix('.txt'), 'w') as f:
                    f.write("1 0.5 0.5 0.9 0.9")  # Dummy: entire image is good
            
            print(f"Processed {len(list(train_dir.glob('*.png')))} good training images")
        
        # Process test set (good + defects)
        test_dir = category_dir / "test"
        good_test_dir = test_dir / "good"
        defect_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        
        img_count = 0
        
        # Good images
        if good_test_dir.exists():
            for img_file in sorted(good_test_dir.glob("*.png"))[:50]:  # Limit to 50 for speed
                dest_img = val_images / img_file.name
                shutil.copy2(img_file, dest_img)
                
                label_file = val_labels / img_file.stem
                with open(label_file.with_suffix('.txt'), 'w') as f:
                    f.write("1 0.5 0.5 0.9 0.9")
                
                img_count += 1
        
        # Defect images with ground truth boxes
        ground_truth_dir = category_dir / "ground_truth"
        for defect_type_dir in defect_dirs:
            for img_file in sorted(defect_type_dir.glob("*.png"))[:20]:  # Limit per type
                # Find corresponding ground truth mask
                defect_type = defect_type_dir.name
                mask_path = ground_truth_dir / defect_type / img_file.name
                
                if mask_path.exists():
                    dest_img = val_images / f"{img_file.stem}_{defect_type}.png"
                    shutil.copy2(img_file, dest_img)
                    
                    # Convert mask to YOLO format
                    yolo_annotation = self.mask_to_yolo_format(str(img_file), str(mask_path))
                    
                    label_file = val_labels / f"{img_file.stem}_{defect_type}"
                    with open(label_file.with_suffix('.txt'), 'w') as f:
                        if yolo_annotation:
                            f.write(yolo_annotation)
                        else:
                            f.write("1 0.5 0.5 0.9 0.9")  # Fallback
                    
                    img_count += 1
        
        print(f"Processed {img_count} validation images")
        
        # Create YOLO dataset.yaml
        dataset_yaml = output_root / "dataset.yaml"
        with open(dataset_yaml, 'w') as f:
            f.write(f"""path: {output_root}
train: images/train
val: images/val

nc: 2
names: ['defect', 'good']
""")
        
        print(f"[OK] Category '{category}' prepared at {output_root}")
        return output_root


def prepare_all_categories(mvtec_root: str = r"C:\Users\User\sentinel\data\mvtec",
                          output_root: str = r"C:\Users\User\sentinel\data\yolo_mvtec"):
    """Prepare all categories to separate directories."""
    preparator = MVTecDatasetPreparator(mvtec_root)
    
    print(f"Found categories: {preparator.categories}")
    
    # Prepare each category to its own directory to avoid file conflicts
    for category in preparator.categories[:5]:  # First 5 categories for testing
        print(f"\nPreparing {category}...")
        category_output = Path(output_root) / category
        preparator.prepare_category(category, str(category_output))
    
    print(f"\n[OK] All categories prepared at {output_root}")


if __name__ == "__main__":
    prepare_all_categories()
