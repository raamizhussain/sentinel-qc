"""
Simple YOLOv10 training pipeline with MLflow tracking.
Directly uses MVTec data without format conversion.
"""

import os
import json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import torch
import mlflow
from datetime import datetime

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = "yolov10-mvtec"

try:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except:
    experiment_id = mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# Start MLflow run
run_timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
run_name = f"yolov10n-bottle-{run_timestamp}"

print(f"Starting MLflow run: {run_name}")
mlflow.start_run(run_name=run_name)

try:
    # Auto-detect device
    device = 0 if torch.cuda.is_available() else "cpu"
    batch_size = 16 if torch.cuda.is_available() else 8  # Reduce batch size for CPU
    
    # Log parameters
    params = {
        "model": "yolov10n",
        "dataset": "bottle",
        "epochs": 5,  # 5 epochs for CPU training
        "batch_size": batch_size,
        "imgsz": 640,
        "device": str(device),
        "patience": 50,
        "save": True,
    }
    
    mlflow.log_params(params)
    
    print(f"Parameters logged to MLflow:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Initialize model
    model_path = Path(__file__).parent / "yolov10n.pt"
    print(f"\nLoading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Log model info
    mlflow.log_param("yolo_version", "yolov10n")
    mlflow.log_param("model_params", str(model.model.parameters() if hasattr(model, 'model') else 'N/A'))
    
    # Prepare dataset YAML for training
    # Create a temporary YAML config for MVTec bottle
    dataset_yaml = Path("../data/mvtec_bottle.yaml")
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    
    # Use absolute paths
    mvtec_bottle_path = Path(__file__).parent.parent / "data" / "mvtec" / "bottle"
    
    yaml_content = f"""
path: {mvtec_bottle_path.absolute()}
train: train
val: test

nc: 2
names: ['defect', 'good']
"""
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created at {dataset_yaml}")
    
    # Train model (just 1 epoch for demonstration)
    print("\nStarting training...")
    print("Note: Using synthetic training (1 epoch) for demonstration")
    print("In production, this would train on full MVTec dataset")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=params["epochs"],
        batch=params["batch_size"],
        imgsz=params["imgsz"],
        device=params["device"],
        patience=params["patience"],
        save=params["save"],
        project=str(Path(__file__).parent.parent / "models"),
        name="yolov10n-bottle"
    )
    
    # Log metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value))
            except:
                pass
    
    # Log model
    models_dir = Path(__file__).parent.parent / "models"
    model_save_path = models_dir / "yolov10n-bottle" / "weights" / "best.pt"
    if model_save_path.exists():
        mlflow.log_artifact(str(model_save_path), "model")
        print(f"\nModel saved and logged to MLflow: {model_save_path}")
    
    # Test on bottle dataset
    print("\nRunning inference on test image...")
    test_img = Path(__file__).parent.parent / "data" / "mvtec" / "bottle" / "test" / "good" / "000.png"
    
    if test_img.exists():
        results = model.predict(source=str(test_img), conf=0.25)
        print(f"Inference completed on {test_img}")
        
        # Log results
        results_dict = {
            "test_image": str(test_img),
            "detections": len(results[0].boxes) if results else 0,
        }
        mlflow.log_param("test_result", json.dumps(results_dict))
    
    mlflow.log_metric("training_complete", 1)
    print("\n[SUCCESS] Training completed and logged to MLflow!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Artifacts URI: {mlflow.get_artifact_uri()}")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    mlflow.log_metric("training_failed", 1)
    mlflow.log_param("error_message", str(e))
    raise

finally:
    mlflow.end_run()
