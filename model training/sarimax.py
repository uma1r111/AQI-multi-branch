import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import json
import os

# MLflow setup
mlflow.set_tracking_uri("http://172.174.154.85:8000")
mlflow.set_experiment("preprod_model_training")

MODEL_NAME = "sarimax-model"  # use the registered name from dev
ALIAS_SRC = "pre-challenger"  # model to evaluate
ALIAS_CURR = "challenger-pre-prod"
ALIAS_BEST = "post-challenger"

# === Load Data ===
df = pd.read_csv("C:/Users/shaikh.mumar/AQI-multi-branch/data/feature_selection.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

train_df = df[(df["datetime"] >= "2025-04-01") & (df["datetime"] <= "2025-07-20")]
test_df = df[(df["datetime"] >= "2025-07-21") & (df["datetime"] <= "2025-07-31")]

target_col = "aqi_us"
test_target = test_df[target_col]
test_exog = test_df.drop(columns=["datetime", target_col]).values

# === Load model from MLflow using alias ===
model_uri = f"models:/{MODEL_NAME}@{ALIAS_SRC}"
model = mlflow.pyfunc.load_model(model_uri=model_uri)

# === Evaluate on test set (internal only) ===
preds = model.predict(test_exog)

# === Metrics ===
rmse = sqrt(mean_squared_error(test_target, preds))
mae = mean_absolute_error(test_target, preds)
mape = np.mean(np.abs((test_target - preds) / test_target)) * 100

print(f"📊 RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

# === Log metrics to MLflow and handle aliasing ===
with mlflow.start_run(run_name="SARIMAX-preprod") as run:
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape})
    run_id = run.info.run_id

    # Save metrics.json for compare_model.py
    with open("metrics.json", "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "mape": mape}, f, indent=4)

    client = MlflowClient()
    version_info = client.get_model_version_by_alias(MODEL_NAME, ALIAS_SRC)
    version = version_info.version

    # Set current pre-prod alias
    client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_CURR, version=version)

    # === Compare to post-challenger ===
    try:
        best_model = client.get_model_version_by_alias(MODEL_NAME, ALIAS_BEST)
        best_rmse = client.get_run(best_model.run_id).data.metrics.get("rmse", float("inf"))
        print(f"📈 Previous post-challenger RMSE: {best_rmse:.4f}")
    except Exception:
        best_model = None
        best_rmse = float("inf")
        print("ℹ️ No existing post-challenger. This may be the first candidate.")

    if rmse < best_rmse:
        if best_model:
            client.delete_registered_model_alias(MODEL_NAME, alias=ALIAS_BEST)
        client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_BEST, version=version)
        print(f"✅ Promoted version {version} to '{ALIAS_BEST}'")
    else:
        print("ℹ️ Model did not outperform current post-challenger.")

