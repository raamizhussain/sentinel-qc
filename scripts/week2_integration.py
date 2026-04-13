"""
Week 2 - Multimodal Defect Analysis Pipeline
Integrates: YOLOv10 (known defects) + PatchCore (unknown) + CLIP (embeddings) + Qdrant (search)
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import mlflow

print("\n" + "="*80)
print("SENTINEL WEEK 2 - MULTIMODAL DEFECT ANALYSIS")
print("="*80 + "\n")

# ============================================================================
# PART 1: Initialize Components
# ============================================================================

print("Initializing components...")

try:
    from patchcore_detector import PatchCoreDetector
    from clip_embedder import CLIPEmbedder
    from qdrant_db import QdrantVectorDB
    
    # Initialize PatchCore
    print("  • PatchCore anomaly detector")
    patchcore = PatchCoreDetector(backbone="resnet50")
    
    # Initialize CLIP
    print("  • CLIP multimodal embedder")
    clip = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
    
    # Initialize Qdrant
    print("  • Qdrant vector database")
    db = QdrantVectorDB(
        collection_name="sentinel-mvtec",
        vector_size=512,
        storage_path="../data/qdrant",
        in_memory=False
    )
    
    print("✓ All components initialized\n")

except Exception as e:
    print(f"✗ Error initializing components: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# PART 2: Build Anomaly Reference Database
# ============================================================================

print("Building reference database on MVTec bottle dataset...")

mvtec_root = Path("../data/mvtec/bottle")

# Collect training images
train_good = list((mvtec_root / "train" / "good").glob("*.png"))[:100]

if train_good:
    print(f"  • Training PatchCore on {len(train_good)} good images")
    patchcore.fit([str(p) for p in train_good], subsample_factor=0.1)
    
    print(f"  • Generating CLIP embeddings for reference set")
    good_embeddings = clip.embed_images([str(p) for p in train_good], batch_size=16)
    
    print(f"  • Uploading to Qdrant ({len(train_good)} images)")
    metadata = [{"category": "bottle", "defect_type": "none", "is_good": True} 
                for _ in train_good]
    db.add_images_batch(good_embeddings, [str(p) for p in train_good], metadata)
    
    print(f"✓ Reference database created\n")

# ============================================================================
# PART 3: Analyze Test Defects
# ============================================================================

print("Testing defect detection on various categories...")

test_categories = {
    "Good": mvtec_root / "test" / "good",
    "Broken Large": mvtec_root / "test" / "broken_large",
    "Broken Small": mvtec_root / "test" / "broken_small",
    "Contamination": mvtec_root / "test" / "contamination",
}

results_all = []

for defect_name, defect_dir in test_categories.items():
    if not defect_dir.exists():
        continue
    
    print(f"\n  {defect_name}:")
    test_images = list(defect_dir.glob("*.png"))[:3]
    
    for img_path in test_images:
        # Read image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. PatchCore anomaly detection
        anomaly_score, anomaly_heatmap = patchcore.detect(image_rgb)
        
        # 2. CLIP embedding
        embedding = clip.embed_image(image_rgb)
        
        # 3. Qdrant similarity search
        similar = db.search(embedding, top_k=3)
        
        # Collect results
        result = {
            "image": str(img_path),
            "category": defect_name,
            "patchcore_anomaly_score": float(anomaly_score),
            "top_similar": [
                {
                    "image": s["image_path"],
                    "similarity": float(s["similarity"]),
                    "defect_type": s["defect_type"]
                }
                for s in similar[:1]  # Top 1
            ]
        }
        
        is_good = defect_name == "Good"
        severity = "Normal" if anomaly_score < 0.3 else ("Medium" if anomaly_score < 0.6 else "High")
        
        print(f"    • {img_path.name}")
        print(f"        Anomaly Score: {anomaly_score:.3f} ({severity})")
        print(f"        Top Match: {Path(similar[0]['image_path']).name} ({similar[0]['similarity']:.3f})")
        
        results_all.append(result)

print(f"\n✓ Analysis complete ({len(results_all)} images processed)\n")

# ============================================================================
# PART 4: Log to MLflow
# ============================================================================

print("Logging Week 2 progress to MLflow...")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("week-2-multimodal")

with mlflow.start_run(run_name="multimodal-pipeline-integration"):
    # Log components
    mlflow.log_param("patchcore_backbone", "resnet50")
    mlflow.log_param("clip_model", "ViT-B-32")
    mlflow.log_param("qdrant_collection", "sentinel-mvtec")
    
    # Log database stats
    stats = db.get_collection_stats()
    mlflow.log_param("qdrant_point_count", stats["vector_count"])
    mlflow.log_param("qdrant_vector_size", stats["vector_size"])
    
    # Log results
    mlflow.log_param("test_images_analyzed", len(results_all))
    
    # Save results JSON
    results_path = Path("../data/week2_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_all, f, indent=2)
    
    mlflow.log_artifact(str(results_path))
    
    print("✓ Results logged to MLflow\n")

# ============================================================================
# PART 5: Summary Report
# ============================================================================

print("="*80)
print("WEEK 2 COMPLETION SUMMARY")
print("="*80)

summary = {
    "Week": 2,
    "Completion_Date": datetime.now().isoformat(),
    "Components_Integrated": [
        "✓ PatchCore anomaly detection (ResNet50 backbone)",
        "✓ CLIP multimodal embeddings (ViT-B-32)",
        "✓ Qdrant vector database (cosine similarity)",
        "✓ Multimodal test pipeline",
    ],
    "Capabilities_Enabled": [
        "Image similarity search within MVTEC bottle category",
        "Unknown anomaly detection via PatchCore",
        "Visual semantic embeddings via CLIP",
        "Batch processing and indexing in Qdrant",
    ],
    "Test_Results": {
        "Good_Images": f"{len([r for r in results_all if r['category'] == 'Good'])} tested",
        "Defect_Images": f"{len([r for r in results_all if r['category'] != 'Good'])} tested",
        "Avg_Anomaly_Score": float(np.mean([r["patchcore_anomaly_score"] for r in results_all])),
    },
    "Database_Status": {
        "Total_Indexed": stats["vector_count"],
        "Vector_Dimension": stats["vector_size"],
        "Storage": "Persistent (data/qdrant)",
    }
}

for key, value in summary.items():
    if isinstance(value, dict):
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    elif isinstance(value, list):
        print(f"\n{key}:")
        for item in value:
            print(f"  {item}")
    else:
        print(f"{key}: {value}")

print("\n" + "="*80)
print("READY FOR WEEK 3 - SYNTHETIC SOPs & TEXT RAG")
print("="*80 + "\n")
