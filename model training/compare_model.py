import json
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# -------------------------------
# 🟠 Constants
# -------------------------------
MODEL_NAME = "sarimax-dev-model"
METRICS_PATH = r'C:\Users\shaikh.mumar\AQI-multi-branch\model training\metrics.json'
CHALLENGER_METRICS_PATH = "challenger_metrics.json"
ALIAS_CHALLENGER = "challenger"
ALIAS_PRE_CHALLENGER = "pre-challenger"

# -------------------------------
# 🟢 MLflow Setup
# -------------------------------
mlflow.set_tracking_uri("http://172.174.154.85:8000")
client = MlflowClient()

# -------------------------------
# 🟡 Input: New Run ID
# -------------------------------
if len(sys.argv) != 2:
    print("Usage: python compare_model.py <run_id>")
    sys.exit(1)

new_run_id = sys.argv[1]

# -------------------------------
# 🔍 Load new model's metrics
# -------------------------------
with open(METRICS_PATH, "r") as f:
    new_metrics = json.load(f)

new_rmse = new_metrics.get("rmse")

# -------------------------------
# 🧠 Get new model version
# -------------------------------
new_model_versions = client.search_model_versions(f"run_id='{new_run_id}'")
if not new_model_versions:
    print(f"❌ No model version found for run_id: {new_run_id}")
    sys.exit(1)

new_model_version = new_model_versions[0].version

# Assign alias 'challenger' to new model (always)
client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_CHALLENGER, version=new_model_version)

# -------------------------------
# 🕵️ Check if pre-challenger exists
# -------------------------------
try:
    pre_challenger_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_PRE_CHALLENGER)
    prev_run_id = pre_challenger_version.run_id
    prev_metrics = client.get_run(prev_run_id).data.metrics
    prev_rmse = prev_metrics.get("rmse", float("inf"))

    print(f"📊 New RMSE: {new_rmse:.4f} | Existing pre-challenger RMSE: {prev_rmse:.4f}")

    if new_rmse < prev_rmse:
        # Remove old alias
        client.delete_registered_model_alias(MODEL_NAME, alias=ALIAS_PRE_CHALLENGER)
        # Assign new alias
        client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_PRE_CHALLENGER, version=new_model_version)
        print(f"✅ Promoted version {new_model_version} as new 'pre-challenger'")
    else:
        print("❌ New model did not outperform existing pre-challenger.")

except MlflowException:
    # No pre-challenger exists → assign alias
    client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_PRE_CHALLENGER, version=new_model_version)
    print(f"🆕 No existing pre-challenger. Assigned new model as 'pre-challenger'")

# -------------------------------
# 💾 Save challenger metrics
# -------------------------------
with open(CHALLENGER_METRICS_PATH, "w") as f:
    json.dump(new_metrics, f, indent=4)

print("✅ Compare process complete.")
