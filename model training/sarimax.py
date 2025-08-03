import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import json
import joblib
import os

# -------------------------------
# MLflow Setup
# -------------------------------
mlflow.set_tracking_uri("http://172.174.154.85:8000")
mlflow.set_experiment("Pre-Prod Model Training")

MODEL_NAME = "aqi-model"
ALIAS_SRC = "pre-challenger"           # 🔹 Best model from Dev phase
ALIAS_CURR = "challenger-pretest"      # 🔹 To be logged here

# -------------------------------
# 🔍 Load SARIMAX Config from Pre-Challenger
# -------------------------------
client = MlflowClient()
try:
    model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_SRC)
except Exception as e:
    raise ValueError(f"❌ Failed to fetch alias '{ALIAS_SRC}' from model registry: {e}")

run_id = model_version.run_id
run = client.get_run(run_id)
params = run.data.params

try:
    p = int(params["p"])
    d = int(params["d"])
    q = int(params["q"])
    P = int(params["P"])
    D = int(params["D"])
    Q = int(params["Q"])
    seasonal_period = int(params["seasonal_period"])
    trend = params.get("trend", "n")
except KeyError as e:
    raise ValueError(f"❌ Missing SARIMAX config in run params: {e}")

order = (p, d, q)
seasonal_order = (P, D, Q, seasonal_period)

print("🔧 Using the following SARIMAX hyperparameters:")
print(f"   • order = {order}")
print(f"   • seasonal_order = {seasonal_order}")
print(f"   • trend = '{trend}'")

# -------------------------------
# Load Pre-Prod Data
# -------------------------------
df = pd.read_csv("C:/Users/shaikh.mumar/AQI-multi-branch/data/feature_selection.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

train_df = df[(df["datetime"] >= "2025-04-01") & (df["datetime"] <= "2025-07-20")]
test_df = df[(df["datetime"] >= "2025-07-21") & (df["datetime"] <= "2025-07-31")]

target_col = "aqi_us"
train_target = train_df[target_col]
test_target = test_df[target_col]

train_exog = train_df.drop(columns=["datetime", target_col]).values
test_exog = test_df.drop(columns=["datetime", target_col]).values

# === Custom MLflow PyFunc Wrapper ===
class SARIMAXWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["sarimax_model"])
    def predict(self, context, model_input):
        return self.model.forecast(steps=len(model_input), exog=model_input)

with mlflow.start_run(run_name="Sarimax-preprod") as run:
    try:
        print("Training SARIMAX...")
        model = SARIMAX(
            endog=train_target,
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=0, maxiter=50)
        preds = fitted_model.forecast(steps=len(test_target), exog=test_exog)

        # === Metrics ===
        rmse = sqrt(mean_squared_error(test_target, preds))
        mae = mean_absolute_error(test_target, preds)
        aic = fitted_model.aic

        print(f"📊 RMSE: {rmse:.4f}, MAE: {mae:.4f}, AIC: {aic:.2f}")

        # === Log Metrics & Params ===
        mlflow.log_params({
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q,
            "seasonal_period": seasonal_period,
            "trend": trend
        })
        mlflow.log_metrics({
            "rmse": rmse, "mae": mae, "aic": aic
        })
        mlflow.set_tag("model_type", "sarimax")

        # === Save and Log Model ===
        joblib.dump(fitted_model, "sarimax_model.pkl")
        mlflow.pyfunc.log_model(
            artifact_path="sarimax_model_pyfunc",
            python_model=SARIMAXWrapper(),
            artifacts={"sarimax_model": "sarimax_model.pkl"}
        )

        # === Register Model ===
        model_uri = f"runs:/{run.info.run_id}/sarimax_model_pyfunc"
        print(f"Model URI: {model_uri}")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS_CURR,
            version=registered_model.version
        )
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=registered_model.version,
            key="model_type",
            value="sarimax"
        )
        print(f"✅ Model registered as version {registered_model.version} with alias '{ALIAS_CURR}'")

        # === Save metrics.json for comparison ===
        with open("metrics.json", "w") as f:
            json.dump({
                "rmse": rmse, "mae": mae, "aic": aic
            }, f, indent=4)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        mlflow.log_param("error", str(e))
