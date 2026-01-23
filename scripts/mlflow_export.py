#!/usr/bin/env python3
"""
MLFlow Export Script for LM-Eval Results

This script exports evaluation results and artifacts to MLFlow tracking server.
It reads the results from the output directory and uploads metrics and artifacts
based on the configuration provided via environment variables.
"""

import os
import sys
import json
import glob
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import mlflow
    import mlflow.client
except ImportError:
    print("ERROR: mlflow package not available. Install with: pip install mlflow-skinny")
    sys.exit(1)


class MLFlowExporter:
    """Handles exporting LM-Eval results to MLFlow tracking server."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
        source_name: Optional[str] = None,
        source_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        artifact_uri: Optional[str] = None,
    ):
        """
        Initialize MLFlow exporter.

        Args:
            tracking_uri: MLFlow tracking server URI
            experiment_name: Name of MLFlow experiment (optional)
            run_id: Specific run ID to use (optional)
            source_name: Value to store in mlflow.source.name tag
            source_type: Value to store in mlflow.source.type tag
            params: Optional dict of parameters to log to MLflow
            artifact_uri: Optional artifact URI to log as a tag (e.g., S3 bucket path)
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or "lm-eval"
        self.run_id = run_id
        self.source_name = source_name
        self.source_type = source_type
        self.params = params or {}
        self.artifact_uri = artifact_uri

        # Configure MLFlow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.client.MlflowClient(tracking_uri=self.tracking_uri)

        # Setup experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name,
                                                            #   artifact_location="oci://1/2"
                                                              )
                self.experiment = mlflow.get_experiment(self.experiment_id)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"ERROR: Failed to setup MLFlow experiment '{self.experiment_name}': {e}")
            raise

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """Make metric names MLflow-safe."""
        return name.replace(",", "_")

    def export_metrics(self, results_data: Dict[str, Any], run_id: Optional[str] = None) -> None:
        """
        Export evaluation metrics to MLFlow.

        Args:
            results_data: Parsed results from LM-Eval JSON output
            run_id: MLFlow run ID to use (optional, uses active run if None)
        """
        if not results_data:
            print("WARNING: No results data provided for metrics export")
            return

        try:
            # Extract metrics from results
            metrics = {}

            # Process task results
            if 'results' in results_data:
                for task_name, task_results in results_data['results'].items():
                    for metric_name, metric_value in task_results.items():
                        if metric_name in {"alias", "samples", " "}:
                            continue
                        if isinstance(metric_value, (int, float)):
                            # Flatten metric name for MLFlow
                            metric_key = f"{task_name}_{self._sanitize_metric_name(metric_name)}"
                            metrics[metric_key] = metric_value
                        elif isinstance(metric_value, dict) and 'metric' in metric_value:
                            # Handle nested metric format
                            metric_key = f"{task_name}_{self._sanitize_metric_name(metric_name)}"
                            metrics[metric_key] = metric_value['metric']

            # Process group results if available
            if 'groups' in results_data:
                for group_name, group_results in results_data['groups'].items():
                    for metric_name, metric_value in group_results.items():
                        if metric_name in {"alias", "samples", " "}:
                            continue
                        if isinstance(metric_value, (int, float)):
                            metric_key = f"group_{group_name}_{self._sanitize_metric_name(metric_name)}"
                            metrics[metric_key] = metric_value

            # Process overall metrics if available
            if 'overall' in results_data:
                for metric_name, metric_value in results_data['overall'].items():
                    if isinstance(metric_value, (int, float)):
                        metric_key = f"overall_{self._sanitize_metric_name(metric_name)}"
                        metrics[metric_key] = metric_value

            # Log metrics to MLFlow (use active run context)
            if metrics:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                print(f"Successfully logged {len(metrics)} metrics to MLFlow")
            else:
                print("WARNING: No numeric metrics found in results data")

        except Exception as e:
            print(f"ERROR: Failed to export metrics to MLFlow: {e}")
            raise

    def export_artifacts(self, output_dir: str, run_id: Optional[str] = None) -> None:
        """
        Export evaluation artifacts to MLFlow.

        Args:
            output_dir: Directory containing evaluation outputs
            run_id: MLFlow run ID to use (optional, uses active run if None)
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"WARNING: Output directory {output_dir} does not exist")
            return

        try:
            # Find artifact files to upload
            artifact_patterns = [
                "*result*.json",     # Result files
                "*.log",             # Log files
                "samples_*.jsonl",   # Sample outputs
                "config.json",       # Configuration files
                "*.txt",             # Text reports
                "*.csv",             # CSV exports
                "*.html",            # HTML reports
            ]

            artifacts_found = set()
            for pattern in artifact_patterns:
                artifacts_found.update(
                    glob.glob(str(output_path / "**" / pattern), recursive=True)
                )

            if artifacts_found:
                for artifact_path in sorted(artifacts_found):
                    # Use the relative directory as the artifact path so we do not create nested duplicate filenames
                    relative_path = os.path.relpath(artifact_path, output_dir)
                    dest_dir = os.path.dirname(relative_path) or None
                    mlflow.log_artifact(artifact_path, artifact_path=dest_dir)

                print(f"Successfully uploaded {len(artifacts_found)} artifacts to MLFlow")
                print(f"Artifacts uploaded: {[os.path.basename(f) for f in artifacts_found]}")
            else:
                print("WARNING: No artifacts found to upload")

        except Exception as e:
            print(f"ERROR: Failed to export artifacts to MLFlow: {e}")
            raise

    def export_results(
        self,
        output_dir: str,
        export_types: List[str],
        results_file: Optional[str] = None,
    ) -> str:
        """
        Export evaluation results to MLFlow based on configuration.

        Args:
            output_dir: Directory containing evaluation outputs
            export_types: List of export types ('metrics', 'artifacts')
            results_file: Specific results file to load (optional)

        Returns:
            MLFlow run ID
        """
        # Find and load results file
        results_data = None
        results_file_path = None

        if results_file:
            provided_path = Path(results_file)
            if provided_path.exists():
                results_file_path = str(provided_path)
            else:
                print(f"WARNING: Specified results file {results_file} does not exist")

        if results_file_path is None:
            results_files = sorted(
                glob.glob(os.path.join(output_dir, "**", "*result*.json"), recursive=True),
                key=os.path.getmtime,
                reverse=True,
            )

            if results_files:
                results_file_path = results_files[0]

        if results_file_path:
            try:
                with open(results_file_path, 'r') as f:
                    results_data = json.load(f)
                print(f"Loaded results from {results_file_path}")
            except Exception as e:
                print(f"ERROR: Failed to load results from {results_file_path}: {e}")
        else:
            print("WARNING: No result files found in output directory")

        # Start MLFlow run
        try:
            with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id) as run:
                run_id = run.info.run_id
                print(f"Started MLFlow run: {run_id}")

                # Log basic run information
                mlflow.set_tag("lm_eval.version", "latest")
                mlflow.set_tag("lm_eval.output_dir", output_dir)
                mlflow.set_tag("prova", "https://google.com")
                mlflow.log_param("OCI Artifact", "quay.io/mmortari/demo20251206:evaluation20251206v2")
                mlflow.set_tag("mlflow.note.content", "Persisted an OCI Artifact for permanent audit of traces and results at: [quay.io/mmortari/demo20251206:evaluation20251206v2](https://quay.io/repository/mmortari/demo20251206?tab=tags&tag=evaluation20251206v2)")
                # md = "[Open Dashboard](https://my.domain.com/page)"
                # path = "link.md"
                # with open(path, "w") as f:
                #     f.write(md)
                # mlflow.log_artifact(path)
                # with open(path, "w") as f:
                #     f.write("overwrite")
                # mlflow.log_artifact(path)
                if self.source_name:
                    mlflow.set_tag("mlflow.source.name", self.source_name)
                if self.source_type:
                    mlflow.set_tag("mlflow.source.type", self.source_type)
                if self.artifact_uri:
                    mlflow.set_tag("mlflow.artifact.uri", self.artifact_uri)
                    print(f"Logged artifact URI: {self.artifact_uri}")
                if self.params:
                    # Convert all values to strings for MLflow params
                    mlflow.log_params(
                        {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in self.params.items()}
                    )

                # Export metrics if requested
                if 'metrics' in export_types and results_data:
                    print("Exporting metrics to MLFlow...")
                    self.export_metrics(results_data, run_id)

                # Export artifacts if requested
                if 'artifacts' in export_types:
                    print("Exporting artifacts to MLFlow...")
                    self.export_artifacts(output_dir, run_id)

                print(f"MLFlow export completed successfully. Run ID: {run_id}")
                print(f"View results at: {self.tracking_uri}/#/experiments/{self.experiment_id}/runs/{run_id}")
                run
                return run_id

        except Exception as e:
            print(f"ERROR: MLFlow export failed: {e}")
            raise


def main():
    """Main entry point for the MLFlow export script."""
    parser = argparse.ArgumentParser(description="Export LM-Eval results to MLFlow")
    parser.add_argument("--output-dir", required=True, help="Directory containing evaluation outputs")
    parser.add_argument("--tracking-uri", help="MLFlow tracking server URI (or set MLFLOW_TRACKING_URI)")
    parser.add_argument("--experiment-name", help="MLFlow experiment name (default: lm-eval)")
    parser.add_argument("--run-id", help="Specific MLFlow run ID to use (optional)")
    parser.add_argument("--results-file", help="Specific results file to load (optional)")
    parser.add_argument("--source-name", help="Value for mlflow.source.name tag (e.g., LMEvalJob CR name)")
    parser.add_argument("--source-type", help="Value for mlflow.source.type tag (e.g., LMEvalJob)")
    parser.add_argument("--artifact-uri", help="Arbitrary artifact URI to log as a tag (e.g., S3 bucket path)")
    parser.add_argument("--params-json", help="JSON dict of parameters to log to MLflow")
    parser.add_argument("--export-types", nargs="+", choices=["metrics", "artifacts"],
                       default=["metrics", "artifacts"], help="What to export to MLFlow")

    args = parser.parse_args()

    # Also check for environment variables (used by the operator)
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = args.experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME")
    run_id = args.run_id or os.getenv("MLFLOW_RUN_ID")
    results_file = args.results_file or os.getenv("MLFLOW_RESULTS_FILE")
    source_name = args.source_name or os.getenv("MLFLOW_SOURCE_NAME")
    source_type = args.source_type or os.getenv("MLFLOW_SOURCE_TYPE")
    artifact_uri = args.artifact_uri or os.getenv("MLFLOW_ARTIFACT_URI")
    params_json = args.params_json or os.getenv("MLFLOW_PARAMS_JSON")
    params = {}
    if params_json:
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            print("WARNING: Could not decode params JSON. Skipping params.")
    export_types_env = os.getenv("MLFLOW_EXPORT_TYPES", "").split(",")
    export_types = args.export_types if args.export_types else [t.strip() for t in export_types_env if t.strip()]

    if not tracking_uri:
        print("ERROR: MLFlow tracking URI is required (--tracking-uri or MLFLOW_TRACKING_URI)")
        sys.exit(1)

    if not export_types:
        export_types = ["metrics", "artifacts"]

    print(f"MLFlow Export Configuration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment Name: {experiment_name or 'lm-eval'}")
    print(f"  Run ID: {run_id or 'auto-generated'}")
    if results_file:
        print(f"  Results File: {results_file}")
    if source_name:
        print(f"  Source Name: {source_name}")
    if source_type:
        print(f"  Source Type: {source_type}")
    if artifact_uri:
        print(f"  Artifact URI: {artifact_uri}")
    if params:
        print(f"  Params: {json.dumps(params)}")
    print(f"  Export Types: {', '.join(export_types)}")
    print(f"  Output Directory: {args.output_dir}")

    try:
        exporter = MLFlowExporter(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_id=run_id,
            source_name=source_name,
            source_type=source_type,
            params=params,
            artifact_uri=artifact_uri,
        )

        final_run_id = exporter.export_results(
            args.output_dir,
            export_types,
            results_file=results_file,
        )
        print(f"\n✅ Export completed successfully!")
        print(f"   Run ID: {final_run_id}")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
