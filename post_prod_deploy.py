import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# -------------------------------
# 🔧 Constants & Configuration
# -------------------------------
MLFLOW_TRACKING_URI = "http://localhost:8000"
EXPERIMENT_NAME = "Post-Prod Model Deployment"  # ✅ Logging promotion here
MODEL_NAME = "aqi-model"

ALIAS_SRC = "challenger-posttest"   # ✅ Model to evaluate and possibly promote
ALIAS_DEST = "champion"             # ✅ Final alias if approved

MIN_ACCEPTABLE_RMSE = 1.0

# -------------------------------
# 🚀 MLflow Setup
# -------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# -------------------------------
# 🔍 Fetch Challenger-Posttest Model
# -------------------------------
try:
    challenger_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_SRC)
except Exception as e:
    raise ValueError(f"❌ Failed to fetch alias '{ALIAS_SRC}': {e}")

challenger_run = client.get_run(challenger_version.run_id)
challenger_rmse = float(challenger_run.data.metrics.get("rmse", -1))
challenger_mae = float(challenger_run.data.metrics.get("mae", -1))
challenger_r2 = float(challenger_run.data.metrics.get("r2", -1))

print(f"📈 Challenger-posttest metrics:")
print(f"   - RMSE: {challenger_rmse}")
print(f"   - MAE : {challenger_mae}")
print(f"   - R2  : {challenger_r2}")

promoted = False

# -------------------------------
# 🔁 Champion Exists? Compare RMSE
# -------------------------------
try:
    champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_DEST)
    champion_run = client.get_run(champion_version.run_id)
    champion_rmse = float(champion_run.data.metrics.get("rmse", -1))

    print(f"\n👑 Existing Champion metrics:")
    print(f"   - RMSE: {champion_rmse}")

    if challenger_rmse < champion_rmse:
        print(f"\n✅ Challenger outperforms Champion ({challenger_rmse} < {champion_rmse}). Promoting Challenger.")
        client.set_registered_model_alias(MODEL_NAME, ALIAS_DEST, challenger_version.version)
        client.delete_registered_model_alias(MODEL_NAME, ALIAS_DEST, champion_version.version)
        promoted = True
    else:
        print(f"\n❌ Challenger did not outperform Champion ({challenger_rmse} ≥ {champion_rmse}). No promotion.")
except RestException:
    print("\n⚠️ No existing champion found. Applying baseline RMSE threshold...")

    if challenger_rmse < MIN_ACCEPTABLE_RMSE:
        print(f"✅ Model passes baseline RMSE check. Promoting to 'champion'.")
        client.set_registered_model_alias(MODEL_NAME, ALIAS_DEST, challenger_version.version)
        promoted = True
    else:
        raise ValueError(f"❌ Model failed RMSE threshold ({challenger_rmse} ≥ {MIN_ACCEPTABLE_RMSE}). Not promoted.")

# -------------------------------
# 📝 Log Promotion in Post-Prod Experiment
# -------------------------------
if promoted:
    with mlflow.start_run(run_name="Sarimax-champion"):
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("promoted_from", ALIAS_SRC)
        mlflow.set_tag("promoted_to", ALIAS_DEST)
        mlflow.set_tag("version_promoted", challenger_version.version)
        mlflow.log_metric("rmse", challenger_rmse)
        mlflow.log_metric("mae", challenger_mae)
        mlflow.log_metric("r2", challenger_r2)
        mlflow.set_tag("status", "Promoted")
        print("\n📦 Promotion event logged to 'Post-Prod Model Deploy' experiment.")
else:
    print("\n🛑 Promotion skipped. No run logged in 'Post-Prod Model Deploy'.")
