"""
MLflow Dashboard - View training experiments and metrics
"""

import os
from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")

try:
    client = MlflowClient(tracking_uri="file:./mlruns")
    
    # Get experiment
    experiments = client.search_experiments()
    
    print("=" * 80)
    print("SENTINEL MLFLOW DASHBOARD")
    print("=" * 80)
    
    for exp in experiments:
        print(f"\n📊 Experiment: {exp.name}")
        print(f"   ID: {exp.experiment_id}")
        print(f"   Status: {exp.lifecycle_stage}")
        
        # Get runs in this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        print(f"   Total Runs: {len(runs)}")
        
        for i, run in enumerate(runs, 1):
            print(f"\n   Run {i}: {run.info.run_name}")
            print(f"   Status: {run.info.status}")
            print(f"   Start Time: {run.info.start_time}")
            print(f"   Duration: {(run.info.end_time - run.info.start_time if run.info.end_time else '...')} ms")
            
            # Parameters
            if run.data.params:
                print(f"   Parameters:")
                for k, v in list(run.data.params.items())[:5]:
                    print(f"     - {k}: {v}")
                if len(run.data.params) > 5:
                    print(f"     ... and {len(run.data.params) - 5} more")
            
            # Metrics
            if run.data.metrics:
                print(f"   Metrics:")
                for k, v in list(run.data.metrics.items())[:5]:
                    print(f"     - {k}: {v}")
                if len(run.data.metrics) > 5:
                    print(f"     ... and {len(run.data.metrics) - 5} more")
            
            # Tags
            if run.data.tags:
                print(f"   Tags:")
                for k, v in list(run.data.tags.items())[:3]:
                    if not k.startswith("mlflow."):
                        print(f"     - {k}: {v}")
            
            # Artifacts
            artifacts = client.list_artifacts(run.info.run_id)
            if artifacts:
                print(f"   Artifacts: {len(artifacts)} item(s)")
                for artifact in artifacts[:3]:
                    print(f"     - {artifact.path}")
    
    print("\n" + "=" * 80)
    print("✅ Dashboard complete!")
    print("=" * 80)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
