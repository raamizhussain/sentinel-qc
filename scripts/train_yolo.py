"""
Train YOLOv10 on MVTec dataset with MLflow tracking.

This script:
1. Prepares MVTec data in YOLO format
2. Trains YOLOv10 with MLflow experiment tracking
3. Logs all hyperparameters, metrics, and model artifacts
4. Evaluates model on validation set
5. Saves best model to models/ folder
"""

import os
import json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import torch
import mlflow
import yaml
from mlflow_utils import MLflowTracker, ExperimentConfig
from prepare_mvtec_data import MVTecDatasetPreparator


def setup_mlflow_experiment(experiment_name: str = "yolov10-mvtec",
                           model_name: str = "yolov10n",
                           dataset: str = "bottle") -> MLflowTracker:
    """Initialize MLflow experiment."""
    tracker = MLflowTracker(
        tracking_uri="file:./mlruns",
        experiment_name=experiment_name
    )
    
    # Start a run
    run_name = f"{model_name}-{dataset}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    tracker.start_run(
        run_name=run_name,
        tags={
            "model": model_name,
            "dataset": dataset,
            "framework": "ultralytics",
            "task": "anomaly-detection"
        }
    )
    
    return tracker


def prepare_training_data(category: str = "bottle") -> Path:
    """Prepare MVTec data in YOLO format."""
    print(f"Preparing {category} dataset...")
    
    preparator = MVTecDatasetPreparator()
    output_path = preparator.prepare_category(
        category,
        output_root=r"C:\Users\User\sentinel\data\yolo_mvtec"
    )
    
    return output_path


def train_yolov10(dataset_yaml: str,
                 model_name: str = "yolov10n",
                 epochs: int = 50,
                 imgsz: int = 640,
                 batch_size: int = 16,
                 patience: int = 20,
                 device: int = 0) -> dict:
    """
    Train YOLOv10 model.
    
    Args:
        dataset_yaml: Path to dataset.yaml
        model_name: YOLOv10 model size
        epochs: Number of training epochs
        imgsz: Image size
        batch_size: Batch size
        patience: Early stopping patience
        device: GPU device ID
        
    Returns:
        Dictionary with training results
    """
    # Load model
    model = YOLO(f"{model_name}.pt")
    
    # Training hyperparameters
    hyperparams = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "patience": patience,
        "device": device,
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005
    }
    
    print(f"Starting training with hyperparameters: {hyperparams}")
    
    # Train model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        device=device,
        project="../models/yolov10_runs",
        name="train",
        exist_ok=True,
        save=True,
        verbose=True
    )
    
    return {
        "hyperparams": hyperparams,
        "results": results,
        "model": model
    }


def evaluate_model(model: YOLO, dataset_yaml: str) -> dict:
    """
    Evaluate trained model.
    
    Args:
        model: Trained YOLO model
        dataset_yaml: Path to dataset.yaml
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating model...")
    
    metrics = model.val(data=dataset_yaml, save=True)
    
    eval_results = {
        "box_loss": float(metrics.box_loss) if metrics.box_loss else 0,
        "cls_loss": float(metrics.cls_loss) if metrics.cls_loss else 0,
        "dfl_loss": float(metrics.dfl_loss) if metrics.dfl_loss else 0,
        "fitness": float(metrics.fitness) if metrics.fitness else 0,
        "map50": float(metrics.map50) if metrics.map50 else 0,
        "map": float(metrics.map) if metrics.map else 0,
    }
    
    return eval_results


def save_model_checkpoint(model: YOLO, dataset: str, tracker: MLflowTracker) -> Path:
    """Save model checkpoint."""
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"yolov10-{dataset}-checkpoint.pt"
    model.save(str(model_path))
    
    print(f"✓ Model saved to {model_path}")
    
    # Log as artifact
    tracker.log_artifact(str(model_path), "models")
    
    return model_path


def log_training_summary(tracker: MLflowTracker,
                        hyperparams: dict,
                        eval_results: dict,
                        dataset: str):
    """Log training summary to MLflow."""
    
    # Log hyperparameters
    tracker.log_params(hyperparams)
    
    # Log metrics
    tracker.log_metrics({
        "validation_" + k: v for k, v in eval_results.items()
    })
    
    # Log summary as artifact
    summary = {
        "dataset": dataset,
        "hyperparameters": hyperparams,
        "evaluation_metrics": eval_results,
        "status": "completed"
    }
    
    tracker.log_dict(summary, "training_summary")
    
    print("✓ Training summary logged to MLflow")


def main(dataset: str = "bottle",
        model_name: str = "yolov10n",
        epochs: int = 50,
        batch_size: int = 16):
    """Main training pipeline."""
    
    print("=" * 60)
    print("SENTINEL - YOLOv10 Training with MLflow Tracking")
    print("=" * 60)
    
    try:
        # Setup MLflow
        tracker = setup_mlflow_experiment(
            model_name=model_name,
            dataset=dataset
        )
        print(f"MLflow run started: {mlflow.active_run().info.run_id}")
        
        # Prepare data
        dataset_path = prepare_training_data(dataset)
        dataset_yaml = dataset_path / "dataset.yaml"
        
        # Verify dataset.yaml exists
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml}")
        
        # Train model
        training_results = train_yolov10(
            dataset_yaml=str(dataset_yaml),
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size
        )
        
        model = training_results["model"]
        hyperparams = training_results["hyperparams"]
        
        # Evaluate model
        eval_results = evaluate_model(model, str(dataset_yaml))
        
        # Save checkpoint
        save_model_checkpoint(model, dataset, tracker)
        
        # Log everything to MLflow
        log_training_summary(tracker, hyperparams, eval_results, dataset)
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print(f"MLflow Experiment: {tracker.experiment_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Artifact URI: {tracker.get_artifact_uri()}")
        print("=" * 60)
        
        # End run
        tracker.end_run()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        if mlflow.active_run():
            mlflow.end_run()
        raise


if __name__ == "__main__":
    # Check GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Run training
    main(
        dataset="bottle",
        model_name="yolov10n",
        epochs=50,
        batch_size=16
    )
