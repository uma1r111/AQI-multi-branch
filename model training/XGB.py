import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from xgboost import XGBRegressor
import mlflow
from mlflow.tracking import MlflowClient
import json

# MLflow setup
mlflow.set_tracking_uri("http://localhost:8000")
mlflow.set_experiment("Dev Model Training")

# === Load Data ===
df = pd.read_csv("data/feature_selection.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

train_df = df[(df["datetime"] >= "2025-04-01") & (df["datetime"] <= "2025-06-30")]
test_df = df[(df["datetime"] >= "2025-07-01") & (df["datetime"] <= "2025-07-10")]

target_col = "aqi_us"
feature_cols = [col for col in df.columns if col not in ["datetime", target_col]]

# === Preprocessing ===
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(train_df[feature_cols])
y_train = scaler_y.fit_transform(train_df[[target_col]]).ravel()

X_test = scaler_x.transform(test_df[feature_cols])
y_test = scaler_y.transform(test_df[[target_col]]).ravel()

# === Best Parameters for XGBoost ===
best_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.0125,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}

with mlflow.start_run(run_name="XGBoost") as run:
    try:
        print("Training XGBoost...")

        model = XGBRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            gamma=best_params['gamma'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            objective='reg:squarederror',
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1))
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        # Metrics
        rmse = sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        mape = np.mean(np.abs((y_true - preds) / y_true)) * 100

        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        mlflow.set_tag("model_type", "xgboost")

        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAPE", mape)

        # Save metrics to JSON
        metrics = {"rmse": rmse, "mae": mae, "mape": mape}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        # Log and register model
        mlflow.sklearn.log_model(model, artifact_path="xgb_model")
        model_uri = f"runs:/{run.info.run_id}/xgb_model"
        result = mlflow.register_model(model_uri=model_uri, name="aqi-model")

        # === Set alias ===
        client = MlflowClient()
        client.set_registered_model_alias(
            name="aqi-model",
            alias="challenger",
            version=result.version
        )

        client.set_model_version_tag(
            name="aqi-model",
            version=result.version,
            key="model_type",
            value="xgboost"
        )

        print(f"✅ Registered XGBoost model as version {result.version} with alias 'challenger'")
        print(f"🏃 View run XGBoost at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        mlflow.log_param("error", str(e))
