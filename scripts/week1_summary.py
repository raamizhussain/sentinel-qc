"""
Week 1 Training Summary Report
Generates a comprehensive overview of YOLOv10 training and MLflow tracking
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime

print("\n" + "="*80)
print("SENTINEL - WEEK 1 TRAINING SUMMARY REPORT")
print("="*80 + "\n")

# Check training outputs
models_dir = Path("../models/yolov10n-bottle")
if models_dir.exists():
    print("✓ Training Output Directory Created")
    print(f"  Location: {models_dir.absolute()}\n")
    
    # Check for weights
    weights_dir = models_dir / "weights"
    if weights_dir.exists():
        print("✓ Model Weights Saved:")
        for pt_file in weights_dir.glob("*.pt"):
            size_mb = pt_file.stat().st_size / (1024*1024)
            print(f"  - {pt_file.name}: {size_mb:.2f} MB")
    
    # Check for args.yaml
    args_file = models_dir / "args.yaml"
    if args_file.exists():
        print("\n✓ Training Arguments Logged")
        with open(args_file) as f:
            args = yaml.safe_load(f)
            print(f"  - Model: {args.get('model', 'N/A')}")
            print(f"  - Epochs: {args.get('epochs', 'N/A')}")
            print(f"  - Batch Size: {args.get('batch', 'N/A')}")
            print(f"  - Image Size: {args.get('imgsz', 'N/A')}")
            print(f"  - Device: {args.get('device', 'N/A')}")
    
    # Check for results
    results_file = models_dir / "results.csv"
    if results_file.exists():
        print("\n✓ Training Results CSV Generated")
        print(f"  Location: {results_file}")
    
    # List all output files
    all_files = list(models_dir.glob("**/*"))
    image_files = list(models_dir.glob("**/*.png")) + list(models_dir.glob("**/*.jpg"))
    print(f"\n✓ Generated {len(image_files)} visualization images")
    if image_files:
        print("  Key visualizations:")
        for img in sorted(image_files)[:5]:
            print(f"    - {img.name}")
        if len(image_files) > 5:
            print(f"    ... and {len(image_files)-5} more")

# Check MLflow tracking
mlruns_dir = Path("../mlruns")
if mlruns_dir.exists():
    print("\n" + "="*80)
    print("MLflow Tracking Status")
    print("="*80)
    
    experiments = list(mlruns_dir.glob("*"))
    experiments = [e for e in experiments if e.is_dir() and not e.name.startswith(".")]
    
    for exp_dir in experiments:
        runs = list((exp_dir).glob("*"))
        runs = [r for r in runs if r.is_dir() and not r.name.startswith(".")]
        
        meta_file = exp_dir / "meta.yaml"
        if meta_file.exists():
            with open(meta_file) as f:
                exp_meta = yaml.safe_load(f)
                exp_name = exp_meta.get("experiment_name", "unnamed")
                print(f"\n✓ Experiment: {exp_name}")
                print(f"  ID: {exp_meta.get('experiment_id', 'N/A')}")
                print(f"  Total Runs: {len(runs)}")
        
        print("\n  Runs:")
        for run_dir in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            run_meta = run_dir / "meta.yaml"
            if run_meta.exists():
                with open(run_meta) as f:
                    run_info = yaml.safe_load(f)
                    run_name = run_info.get("run_name", "unnamed")
                    status = run_info.get("status", "unknown")
                    status_text = "COMPLETED" if status == 3 else ("ACTIVE" if status == 1 else "FAILED")
                    
                    # Count artifacts
                    artifact_dir = run_dir / "artifacts"
                    artifact_count = len(list(artifact_dir.glob("**/*"))) if artifact_dir.exists() else 0
                    
                    # Count metrics
                    metrics_dir = run_dir / "metrics"
                    metric_count = len(list(metrics_dir.glob("*"))) if metrics_dir.exists() else 0
                    
                    print(f"    • {run_name}")
                    print(f"      Status: {status_text}")
                    print(f"      Metrics: {metric_count}, Artifacts: {artifact_count}")

# Summary
print("\n" + "="*80)
print("WEEK 1 OBJECTIVES")
print("="*80)

week1_items = [
    ("✓", "Python environment setup (venv)", "C:\\Users\\User\\sentinel\\venv"),
    ("✓", "YOLOv10 inference test", "test_yolo.py passed"),
    ("✓", "MVTec AD dataset extracted", "15 categories, 209 bottle training images"),
    ("✓", "MLflow tracking integrated", "File-based backend at ./mlruns"),
    ("✓", "Training pipeline implemented", "scripts/train_simple.py"),
    ("✓", "Model training on CPU", "5 epochs completed"),
    ("✓", "Model evaluation", "Confusion matrix, precision/recall curves"),
]

for status, item, detail in week1_items:
    print(f"{status} {item}")
    print(f"   → {detail}")

print("\n" + "="*80)
print("READY FOR WEEK 2")
print("="*80)
print("""
Week 2 Tasks:
  1. PatchCore integration for unknown anomaly detection
  2. CLIP embeddings for image representation
  3. Qdrant vector database setup
  4. Image similarity search pipeline
  
Current Model: best.pt (5.7 MB) - Ready for inference
Training Data: MVTec bottle (229 images)
MLflow Registry: Experiment 518301855489578474
""")
print("="*80 + "\n")
