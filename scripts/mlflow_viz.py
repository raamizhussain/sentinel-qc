"""MLflow visualization and experiment tracking utilities."""

import pandas as pd
import mlflow
from mlflow_utils import MLflowTracker
from pathlib import Path


def list_experiments(tracking_uri: str = "file:./mlruns"):
    """List all MLflow experiments."""
    mlflow.set_tracking_uri(tracking_uri)
    experiments = mlflow.search_experiments()
    
    print("\n" + "=" * 80)
    print("MLflow Experiments")
    print("=" * 80)
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Runs: {len(runs)}")
        
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            print(f"  Latest Run: {latest_run['run_id']}")
            print(f"  Status: {latest_run['status']}")
            print(f"  Start Time: {latest_run['start_time']}")
    
    print("\n" + "=" * 80 + "\n")


def show_run_summary(experiment_name: str = "yolov10-mvtec", 
                     run_number: int = -1,
                     tracking_uri: str = "file:./mlruns"):
    """Display summary of a specific run."""
    mlflow.set_tracking_uri(tracking_uri)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    
    if len(runs) == 0:
        print(f"No runs found in experiment '{experiment_name}'")
        return
    
    # Get the specified run
    run = runs.iloc[run_number]
    
    print("\n" + "=" * 80)
    print(f"Run Summary: {run['run_id']}")
    print("=" * 80)
    
    print(f"\nMetadata:")
    print(f"  Status: {run['status']}")
    print(f"  Start Time: {run['start_time']}")
    print(f"  Duration: {run.get('duration', 'N/A')} ms")
    
    # Extract metrics
    metric_cols = [col for col in run.index if col.startswith('metrics.')]
    if metric_cols:
        print(f"\nMetrics:")
        for col in metric_cols:
            metric_name = col.replace('metrics.', '')
            value = run[col]
            print(f"  {metric_name}: {value}")
    
    # Extract parameters
    param_cols = [col for col in run.index if col.startswith('params.')]
    if param_cols:
        print(f"\nParameters:")
        for col in param_cols:
            param_name = col.replace('params.', '')
            value = run[col]
            print(f"  {param_name}: {value}")
    
    # Extract tags
    tag_cols = [col for col in run.index if col.startswith('tags.')]
    if tag_cols:
        print(f"\nTags:")
        for col in tag_cols:
            tag_name = col.replace('tags.', '')
            value = run[col]
            print(f"  {tag_name}: {value}")
    
    print("\n" + "=" * 80 + "\n")


def compare_runs(experiment_name: str = "yolov10-mvtec",
                 tracking_uri: str = "file:./mlruns"):
    """Compare all runs in an experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    
    if len(runs) == 0:
        print(f"No runs found in experiment '{experiment_name}'")
        return
    
    print("\n" + "=" * 80)
    print(f"Comparing {len(runs)} runs in experiment '{experiment_name}'")
    print("=" * 80 + "\n")
    
    # Select key metrics and parameters for comparison
    display_cols = ['run_id', 'status', 'start_time']
    
    # Add metric columns
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    display_cols.extend(metric_cols[:5])  # Top 5 metrics
    
    # Add parameter columns
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    display_cols.extend(param_cols[:3])  # Top 3 params
    
    # Create comparison dataframe
    comparison = runs[[col for col in display_cols if col in runs.columns]].copy()
    
    # Rename columns for readability
    comparison.columns = [col.replace('metrics.', 'm_').replace('params.', 'p_') 
                         for col in comparison.columns]
    
    print(comparison.to_string())
    print("\n" + "=" * 80 + "\n")


def export_run_artifacts(experiment_name: str = "yolov10-mvtec",
                        run_number: int = -1,
                        output_dir: str = "./exported_runs",
                        tracking_uri: str = "file:./mlruns"):
    """Export a run's artifacts to local directory."""
    mlflow.set_tracking_uri(tracking_uri)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if len(runs) == 0:
        print(f"No runs found in experiment '{experiment_name}'")
        return
    
    run = runs.iloc[run_number]
    run_id = run['run_id']
    
    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download all artifacts
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    
    print(f"\nExporting artifacts from run {run_id} to {output_path}...")
    
    for artifact in artifacts:
        if artifact.is_dir:
            # Handle directories
            dir_path = output_path / artifact.path
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            # Handle files
            local_path = client.download_artifacts(run_id, artifact.path, str(output_path))
            print(f"  ✓ {artifact.path}")
    
    print(f"\n✓ Artifacts exported to {output_path}\n")


if __name__ == "__main__":
    # Example usage
    list_experiments()
    show_run_summary(run_number=-1)
    compare_runs()
