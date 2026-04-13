"""
Week 2 Component Test - Tests each component independently
Simplified for CPU-only environment
"""

import os
os.environ['CLIP_VIT_L_14_OPENAI_SHA256'] = '7e570f76412369c6a9af4a9b737d00374ddd3718'  # Fix CLIP model hash

import sys
import numpy as np
from pathlib import Path
import json

print("\n=== WEEK 2 COMPONENT TEST ===\n")

# Test 1: PatchCore
print("TEST 1: PatchCore Anomaly Detector")
try:
    from patchcore_detector import PatchCoreDetector
    import cv2
    
    detector = PatchCoreDetector(backbone="resnet18")  # Use smaller model for CPU
    print("  ✓ PatchCore initialized (ResNet18)")
    
    # Test on single image
    test_img = Path("../data/mvtec/bottle/test/good/000.png")
    if test_img.exists():
        image = cv2.imread(str(test_img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Fit quickly
        train_imgs = list(Path("../data/mvtec/bottle/train/good").glob("*.png"))[:5]
        detector.fit([str(p) for p in train_imgs])
        print(f"  ✓ Trained on {len(train_imgs)} images")
        
        # Detect
        score, heatmap = detector.detect(image)
        print(f"  ✓ Detection: anomaly_score={score:.3f}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: CLIP (might be slow, skip if taking too long)
print("\nTEST 2: CLIP Embedder")
try:
    print("  ⏳ Loading CLIP (this may take a minute on CPU)...")
    from clip_embedder import CLIPEmbedder
    import cv2
    
    embedder = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
    print("  ✓ CLIP initialized")
    
    # Test image embedding
    test_img = Path("../data/mvtec/bottle/test/good/000.png")
    if test_img.exists():
        embedding = embedder.embed_image(str(test_img))
        print(f"  ✓ Image embedding shape: {embedding.shape}")
    
    # Test text embedding
    text_embedding = embedder.embed_text("a defective bottle")
    print(f"  ✓ Text embedding shape: {text_embedding.shape}")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Qdrant
print("\nTEST 3: Qdrant Vector Database")
try:
    from qdrant_db import QdrantVectorDB
    
    db = QdrantVectorDB(collection_name="sentinel-test", vector_size=512, in_memory=True)
    print("  ✓ Qdrant initialized (in-memory)")
    
    # Add test data
    embeddings = np.random.randn(5, 512).astype(np.float32)
    paths = [f"img_{i}.png" for i in range(5)]
    metadata = [{"category": "bottle", "is_good": i < 3} for i in range(5)]
    
    db.add_images_batch(embeddings, paths, metadata)
    print(f"  ✓ Added 5 test embeddings")
    
    # Search
    query = embeddings[0]
    results = db.search(query, top_k=2)
    print(f"  ✓ Search returned {len(results)} results")
    
    # Stats
    stats = db.get_collection_stats()
    print(f"  ✓ Database stats: {stats['vector_count']} points")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TEST COMPLETE ===\n")
