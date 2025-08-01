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

# MLflow setup
mlflow.set_tracking_uri("http://172.174.154.85:8000")
mlflow.set_experiment("preprod_model_training")

MODEL_NAME = "sarimax-preprod-model"
ALIAS_BEST = "post-challenger"
ALIAS_LATEST = "challenger-pre-prod"

# === Load Data ===
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

best_params = {
    'p': 2, 'd': 0, 'q': 0,
    'P': 2, 'D': 0, 'Q': 2,
    'seasonal_period': 12
}

# === Custom MLflow PyFunc Wrapper ===
class SARIMAXWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["sarimax_model"])

    def predict(self, context, model_input):
        return self.model.forecast(steps=len(model_input), exog=model_input)

with mlflow.start_run(run_name="SARIMAX-preprod") as run:
    try:
        print("Training SARIMAX (Pre-Prod)...")
        model = SARIMAX(
            endog=train_target,
            exog=train_exog,
            order=(best_params["p"], best_params["d"], best_params["q"]),
            seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], best_params["seasonal_period"]),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=0, maxiter=50)
        preds = fitted_model.forecast(steps=len(test_target), exog=test_exog)

        # === Metrics ===
        rmse = sqrt(mean_squared_error(test_target, preds))
        mae = mean_absolute_error(test_target, preds)
        mape = np.mean(np.abs((test_target - preds) / test_target)) * 100
        aic = fitted_model.aic
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, AIC: {aic:.2f}")

        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape, "aic": aic})

        # Save model
        joblib.dump(fitted_model, "sarimax_model.pkl")
        mlflow.pyfunc.log_model(
            artifact_path="sarimax_model_pyfunc",
            python_model=SARIMAXWrapper(),
            artifacts={"sarimax_model": "sarimax_model.pkl"}
        )

        model_uri = f"runs:/{run.info.run_id}/sarimax_model_pyfunc"
        print(f"Model URI: {model_uri}")

        # Register model
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        version = registered_model.version

        client = MlflowClient()

        # === Fetch existing post-challenger for comparison ===
        try:
            best_model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_BEST)
            prev_metrics = client.get_run(best_model_version.run_id).data.metrics
            prev_rmse = prev_metrics.get("rmse", float("inf"))
            print(f"Previous post-challenger RMSE: {prev_rmse:.4f}")
        except Exception:
            best_model_version = None
            prev_rmse = float("inf")
            print("No post-challenger model found — assigning new model as post-challenger.")

        # === Aliasing logic ===
        client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_LATEST, version=version)

        if rmse < prev_rmse:
            if best_model_version:
                client.delete_registered_model_alias(MODEL_NAME, alias=ALIAS_BEST)
            client.set_registered_model_alias(MODEL_NAME, alias=ALIAS_BEST, version=version)
            print(f"✅ New model outperformed. Assigned aliases: '{ALIAS_LATEST}', '{ALIAS_BEST}'")
        else:
            print(f"ℹ️ Model did not outperform. Assigned alias: '{ALIAS_LATEST}' only")

        # Save metrics for future comparison
        with open("metrics.json", "w") as f:
            json.dump({"rmse": rmse, "mae": mae, "mape": mape, "aic": aic}, f, indent=4)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        mlflow.log_param("error", str(e))
