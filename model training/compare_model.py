import json
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# -------------------------------
# 🔧 Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Compare MLflow models by metrics and aliases.")
parser.add_argument("--model_name", required=True, help="Registered MLflow model name")
parser.add_argument("--new_alias", default="challenger", help="Alias for new model (default: challenger)")
parser.add_argument("--prev_alias", default="pre-challenger", help="Alias for previous model (default: pre-challenger)")
parser.add_argument("--metrics_path", default="metrics.json", help="Path to JSON file with new model's metrics")
parser.add_argument("--challenger_metrics_path", default="challenger_metrics.json", help="Path to JSON file with previous model's metrics")
args = parser.parse_args()

# -------------------------------
# 🔗 MLflow Setup
# -------------------------------
mlflow.set_tracking_uri("http://172.174.154.85:8000")
client = MlflowClient()

# -------------------------------
# Constants (from CLI)
# -------------------------------
NEW_METRICS_PATH = args.metrics_path
CHALLENGER_METRICS_PATH = args.challenger_metrics_path
MODEL_NAME = args.model_name
NEW_ALIAS = args.new_alias
PREVIOUS_ALIAS = args.prev_alias


# === Sanity Check: List All Registered Models ===
try:
    registered_models = client.search_registered_models()
    print(f"🔍 Found {len(registered_models)} registered models:")
    for model in registered_models:
        print(f"  - {model.name}")
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            aliases = getattr(version, 'aliases', [])
            print(f"    Version {version.version}: aliases = {aliases}")
except Exception as e:
    print(f"❌ Failed to connect to MLflow: {e}")
    sys.exit(1)

# === Load New Model Metrics ===
try:
    with open(NEW_METRICS_PATH, "r") as f:
        new_metrics = json.load(f)
    new_rmse = new_metrics.get("rmse")
except Exception as e:
    print(f"❌ Failed to load new metrics: {e}")
    sys.exit(1)

# === First-Time Challenger Promotion (If No Previous Exists) ===
if not os.path.exists(CHALLENGER_METRICS_PATH):
    print("📢 No previous challenger found. Promoting current model as the new challenger.")
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, PREVIOUS_ALIAS)
        version = alias_info.version
        client.set_registered_model_alias(MODEL_NAME, NEW_ALIAS, version=version)

        with open(CHALLENGER_METRICS_PATH, "w") as f:
            json.dump(new_metrics, f, indent=4)

        print(f"✅ Version {version} promoted to alias '{NEW_ALIAS}'")
        sys.exit(0)

    except MlflowException as e:
        print(f"❌ Could not find alias '{PREVIOUS_ALIAS}' or model '{MODEL_NAME}'.")
        print(f"Details: {e}")
        sys.exit(1)

# === Load Current Challenger Metrics ===
try:
    with open(CHALLENGER_METRICS_PATH, "r") as f:
        challenger_metrics = json.load(f)
    challenger_rmse = challenger_metrics.get("rmse")
except Exception as e:
    print(f"❌ Failed to load challenger metrics: {e}")
    sys.exit(1)

# === Compare Metrics ===
print(f"📊 New RMSE: {new_rmse:.4f} | Current Challenger RMSE: {challenger_rmse:.4f}")

if new_rmse < challenger_rmse:
    print("✅ New model outperforms the challenger. Promoting...")

    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, PREVIOUS_ALIAS)
        version = alias_info.version
        client.set_registered_model_alias(MODEL_NAME, NEW_ALIAS, version=version)

        with open(CHALLENGER_METRICS_PATH, "w") as f:
            json.dump(new_metrics, f, indent=4)

        print(f"🏆 Version {version} is now the new challenger.")
        sys.exit(0)

    except MlflowException as e:
        print(f"❌ Failed to promote model: {e}")
        sys.exit(1)

else:
    print("❌ New model did not outperform the current challenger.")
    sys.exit(1)