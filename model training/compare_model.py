import json
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# -------------------------------
# 🟠 Constants
# -------------------------------
MLFLOW_TRACKING_URI = "http://localhost:8000"
EXPERIMENT_NAME = "Pre-Prod Model Training"
METRICS_PATH = "metrics.json"
CHALLENGER_METRICS_PATH = "challenger-posttest_metrics.json"
MODEL_NAME = "aqi-model"
ALIAS_CHALLENGER = "challenger-pretest"
ALIAS_PRE_CHALLENGER = "challenger-posttest"

# -------------------------------
# 🟢 MLflow Setup
# -------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# -------------------------------
# 🧠 Get latest successful run from pre-prod experiment
# -------------------------------
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
    sys.exit(1)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)

if not runs:
    print(f"❌ No runs found in experiment '{EXPERIMENT_NAME}'")
    sys.exit(1)

latest_run = runs[0]
run_id = latest_run.info.run_id
print(f"🆕 Using latest run_id: {run_id}")

# -------------------------------
# 📊 Load new model's metrics
# -------------------------------
if not os.path.exists(METRICS_PATH):
    print(f"❌ Could not find {METRICS_PATH}. Did training run save it?")
    sys.exit(1)

with open(METRICS_PATH, "r") as f:
    new_metrics = json.load(f)

new_rmse = new_metrics.get("rmse")
if new_rmse is None:
    print("❌ RMSE not found in new model metrics.")
    sys.exit(1)

# -------------------------------
# 🔍 Get model version from run_id
# -------------------------------
new_model_versions = client.search_model_versions(f"run_id='{run_id}' and name='{MODEL_NAME}'")
if not new_model_versions:
    print(f"❌ No model version found for run_id: {run_id}")
    sys.exit(1)

new_model_version = new_model_versions[0].version

# -------------------------------
# 🔄 Assign 'challenger-pretest' alias to new model version
# -------------------------------
try:
    old_challenger_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHALLENGER)
    if old_challenger_version.version != new_model_version:
        client.delete_registered_model_alias(MODEL_NAME, alias=ALIAS_CHALLENGER)
except MlflowException:
    pass  # No previous alias

client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_CHALLENGER, version=new_model_version)
print(f"✅ Assigned alias '{ALIAS_CHALLENGER}' to model version {new_model_version}")

# -------------------------------
# 🔍 Compare against posttest model (if exists)
# -------------------------------
promoted = False

try:
    pre_challenger_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_PRE_CHALLENGER)
    prev_run_id = pre_challenger_version.run_id

    # Prevent self-comparison
    if run_id == prev_run_id:
        print("⚠️ Challenger and pre-challenger are the same model. Skipping comparison.")
        sys.exit(0)

    prev_metrics = client.get_run(prev_run_id).data.metrics
    prev_rmse = prev_metrics.get("rmse", float("inf"))

    print(f"📊 New RMSE: {new_rmse:.4f} | Existing posttest RMSE: {prev_rmse:.4f}")

    if new_rmse < prev_rmse:
        client.delete_registered_model_alias(MODEL_NAME, alias=ALIAS_PRE_CHALLENGER)
        client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_PRE_CHALLENGER, version=new_model_version)
        print(f"✅ New model promoted as '{ALIAS_PRE_CHALLENGER}'")
        promoted = True
    else:
        print(f"❌ New model did not outperform current '{ALIAS_PRE_CHALLENGER}'")

except MlflowException:
    # No posttest model exists — use threshold strategy
    if new_rmse < 1.0:
        client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_PRE_CHALLENGER, version=new_model_version)
        print(f"🆕 No previous '{ALIAS_PRE_CHALLENGER}' found. Promoted model based on RMSE < 1.0")
        promoted = True
    else:
        print(f"❌ No previous posttest model, but RMSE >= 1.0 — not promoting.")

# -------------------------------
# 🧹 Remove 'pretest' alias if model is promoted to posttest
# -------------------------------
if promoted:
    try:
        client.delete_registered_model_alias(MODEL_NAME, alias=ALIAS_CHALLENGER)
        print(f"🧹 Removed alias '{ALIAS_CHALLENGER}' after promotion to posttest.")
    except MlflowException:
        print(f"⚠️ Could not remove alias '{ALIAS_CHALLENGER}' — may not exist.")

# -------------------------------
# 💾 Save challenger metrics
# -------------------------------
with open(CHALLENGER_METRICS_PATH, "w") as f:
    json.dump(new_metrics, f, indent=4)

print("✅ Compare process complete.")
