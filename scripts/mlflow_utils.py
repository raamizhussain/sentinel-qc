"""MLflow utilities for experiment tracking and model management."""

import mlflow
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json


class MLflowTracker:
    """Wrapper for MLflow operations with sensible defaults for Sentinel project."""
    
    def __init__(self, tracking_uri: str = "file:./mlruns", experiment_name: str = "sentinel"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: Path to MLflow backend (default: local file)
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking server
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        try:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a model artifact."""
        mlflow.pytorch.log_model(model, artifact_path, **kwargs) if hasattr(model, 'parameters') \
            else mlflow.log_artifact(str(model), artifact_path)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log a file or directory as artifact."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, data: Dict[str, Any], name: str):
        """Log a dictionary as JSON artifact."""
        artifact_dir = mlflow.get_artifact_uri()
        os.makedirs(artifact_dir, exist_ok=True)
        
        json_path = Path(artifact_dir) / f"{name}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_artifact_uri(self) -> str:
        """Get current artifact URI."""
        return mlflow.get_artifact_uri()
    
    def search_runs(self, experiment_ids: list = None, filter_string: str = None) -> list:
        """Search for runs in experiment."""
        return mlflow.search_runs(
            experiment_ids=experiment_ids or [self.experiment_id],
            filter_string=filter_string
        )
    
    def get_best_run(self, metric_name: str, mode: str = "max"):
        """Get the best run by a metric."""
        runs = self.search_runs()
        if not runs.empty:
            if mode == "max":
                best_idx = runs[f"metrics.{metric_name}"].idxmax()
            else:
                best_idx = runs[f"metrics.{metric_name}"].idxmin()
            return runs.loc[best_idx]
        return None


class ExperimentConfig:
    """Configuration for an experiment run."""
    
    def __init__(self):
        self.hyperparams = {}
        self.data_config = {}
        self.model_config = {}
        self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hyperparams": self.hyperparams,
            "data_config": self.data_config,
            "model_config": self.model_config,
            "tags": self.tags
        }
    
    def from_dict(self, config: Dict[str, Any]):
        """Load config from dictionary."""
        self.hyperparams = config.get("hyperparams", {})
        self.data_config = config.get("data_config", {})
        self.model_config = config.get("model_config", {})
        self.tags = config.get("tags", {})
