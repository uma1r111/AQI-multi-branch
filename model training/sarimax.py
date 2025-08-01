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
mlflow.set_experiment("Dev Model Training")

# === Load Data ===
df = pd.read_csv("C:/Users/shaikh.mumar/AQI-multi-branch/data/feature_selection.csv")

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

train_df = df[(df["datetime"] >= "2025-04-01") & (df["datetime"] <= "2025-06-30")]
test_df = df[(df["datetime"] >= "2025-07-01") & (df["datetime"] <= "2025-07-10")]

target_col = "aqi_us"

train_target = train_df[target_col]
test_target = test_df[target_col]

train_exog = train_df.drop(columns=["datetime", target_col]).values
test_exog = test_df.drop(columns=["datetime", target_col]).values

best_params = {
    'p': 2,
    'd': 0, 
    'q': 0,
    'P': 2, 
    'D': 0, 
    'Q': 2,
    'seasonal_period': 12
}

# best_params = {
#     "p": 0,
#     "d": 0,
#     "q": 0,
#     "P": 2,
#     "D": 0,
#     "Q": 2,
#     "seasonal_period": 6
# }

# === Custom MLflow PyFunc Wrapper ===
class SARIMAXWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["sarimax_model"])
    def predict(self, context, model_input):
        return self.model.forecast(steps=len(model_input), exog=model_input)
with mlflow.start_run(run_name="SARIMAX-challenger") as run:
    try:
        print("Training SARIMAX...")
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
        # === Log Metrics & Params ===
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "rmse": rmse, "mae": mae, "mape": mape, "aic": aic
        })

        mlflow.set_tag("model_type", "sarimax")

        # === Save model ===
        joblib.dump(fitted_model, "sarimax_model.pkl")
        mlflow.pyfunc.log_model(
            artifact_path="sarimax_model_pyfunc",
            python_model=SARIMAXWrapper(),
            artifacts={"sarimax_model": "sarimax_model.pkl"}
        )

        # === Register model ===
        model_uri = f"runs:/{run.info.run_id}/sarimax_model_pyfunc"
        print(f"Model URI: {model_uri}")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="aqi-model"
        )
        # Set alias
        client = MlflowClient()
        client.set_registered_model_alias(
            name="aqi-model",
            alias="challenger",
            version=registered_model.version
        )
        print(f"✅ Model registered as version {registered_model.version} with alias 'challenger'")
        # Save metrics.json
        with open("metrics.json", "w") as f:
            json.dump({
                "rmse": rmse, "mae": mae, "mape": mape, "aic": aic
            }, f, indent=4)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        mlflow.log_param("error", str(e))