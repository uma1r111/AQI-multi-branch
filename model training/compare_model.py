import json
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# MLflow setup - THIS WAS MISSING!
mlflow.set_tracking_uri("http://172.174.154.85:8000")

# Constants
NEW_METRICS_PATH = r'C:\Users\shaikh.mumar\AQI-multi-branch\model training\metrics.json'
CHALLENGER_METRICS_PATH = "challenger_metrics.json"
MODEL_NAME = "sarimax-model"
NEW_ALIAS = "challenger"
PREVIOUS_ALIAS = "pre-challenger"

# Initialize MLflow client AFTER setting tracking URI
client = MlflowClient()

# Debug: List all registered models to verify connection
try:
    registered_models = client.search_registered_models()
    print(f"🔍 Found {len(registered_models)} registered models:")
    for model in registered_models:
        print(f"  - {model.name}")
        # Get model versions and aliases
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            aliases = getattr(version, 'aliases', [])
            print(f"    Version {version.version}: aliases = {aliases}")
except Exception as e:
    print(f"❌ Failed to connect to MLflow: {e}")
    sys.exit(1)

# Load new model's metrics
with open(NEW_METRICS_PATH, "r") as f:
    new_metrics = json.load(f)

# If no previous challenger exists
if not os.path.exists(CHALLENGER_METRICS_PATH):
    print("No previous challenger found. Promoting current model as new challenger.")
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, PREVIOUS_ALIAS)
        latest_version = alias_info.version
        client.set_registered_model_alias(MODEL_NAME, NEW_ALIAS, version=latest_version)
        print(f"✅ Promoted version {latest_version} to alias '{NEW_ALIAS}'")
        
        with open(CHALLENGER_METRICS_PATH, "w") as f:
            json.dump(new_metrics, f, indent=4)
        sys.exit(0)
        
    except MlflowException as e:
        print(f"❌ Could not find alias '{PREVIOUS_ALIAS}' or model '{MODEL_NAME}' in registry.")
        print(f"Details: {e}")
        sys.exit(1)

# Load previous challenger metrics
with open(CHALLENGER_METRICS_PATH, "r") as f:
    challenger_metrics = json.load(f)

# Compare RMSE
new_rmse = new_metrics.get("rmse")
challenger_rmse = challenger_metrics.get("rmse")
print(f"📊 New RMSE: {new_rmse:.4f} | Current Challenger RMSE: {challenger_rmse:.4f}")

if new_rmse < challenger_rmse:
    print("✅ New model outperforms the challenger. Promoting...")
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, PREVIOUS_ALIAS)
        latest_version = alias_info.version
        client.set_registered_model_alias(MODEL_NAME, NEW_ALIAS, version=latest_version)
        
        with open(CHALLENGER_METRICS_PATH, "w") as f:
            json.dump(new_metrics, f, indent=4)
        print(f"🏆 Version {latest_version} is now the new challenger.")
        sys.exit(0)
        
    except MlflowException as e:
        print(f"❌ Failed to promote model: {e}")
        sys.exit(1)
else:
    print("❌ New model did not outperform the challenger.")
    sys.exit(1)