import os
import json
import joblib
import pandas as pd

from sqlalchemy import create_engine

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

DATABASE_URL = "sqlite:///./voltedge.db"
MODEL_DIR = "models_ml"

os.makedirs(MODEL_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL)

model_report = {}


def save_report():
    with open(os.path.join(MODEL_DIR, "model_report.json"), "w", encoding="utf-8") as file:
        json.dump(model_report, file, indent=4)


def regression_metrics(name, y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 50)
    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 3))

    model_report[name] = {
        "type": "regression",
        "mae": round(mae, 3),
        "mse": round(mse, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
    }


def print_feature_importance(name, model, features):
    importance = {}

    print("\nFeature importance")
    print("-" * 50)

    for feature, value in zip(features, model.feature_importances_):
        importance[feature] = round(float(value), 4)
        print(f"{feature:<25} {value:.4f}")

    model_report[name]["feature_importance"] = importance


def load_data():
    data = {
        "kunder": pd.read_sql("SELECT * FROM kunder", engine),
        "elbiler": pd.read_sql("SELECT * FROM elbiler", engine),
        "ladestationer": pd.read_sql("SELECT * FROM ladestationer", engine),
        "ladepunkter": pd.read_sql("SELECT * FROM ladepunkter", engine),
        "tariffer": pd.read_sql("SELECT * FROM tariffer", engine),
        "opladninger": pd.read_sql("SELECT * FROM opladninger", engine),
        "driftstilstande": pd.read_sql("SELECT * FROM driftstilstande", engine),
        "servicehaendelser": pd.read_sql("SELECT * FROM servicehaendelser", engine),
    }

    print("Loaded data")
    print("-" * 50)

    for name, df in data.items():
        print(f"{name:<20} {len(df)} rows")

    return data


def add_charger_age(drift_df, ladepunkter_df):
    df = drift_df.merge(
        ladepunkter_df[["ladepunkt_id", "installationsdato", "maks_effekt_kw"]],
        on="ladepunkt_id",
        how="left",
    )

    df["maaletidspunkt"] = pd.to_datetime(df["maaletidspunkt"])
    df["installationsdato"] = pd.to_datetime(df["installationsdato"])

    df["charger_age_days"] = (
        df["maaletidspunkt"] - df["installationsdato"]
    ).dt.days.clip(lower=0)

    return df


def train_health_score_model(data):
    df = add_charger_age(
        data["driftstilstande"],
        data["ladepunkter"]
    ).dropna()

    features = [
        "temperatur_c",
        "spaending_v",
        "stroem_a",
        "fejl_antal",
        "oppetid_procent",
        "maks_effekt_kw",
        "charger_age_days",
    ]

    target = "driftsscore"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    name = "Model 1 - Charger Health Score Prediction"
    regression_metrics(name, y_test, predictions)
    print_feature_importance(name, model, features)

    joblib.dump(model, os.path.join(MODEL_DIR, "health_score_model.pkl"))


def train_anomaly_detection_model(data):
    df = add_charger_age(
        data["driftstilstande"],
        data["ladepunkter"]
    ).dropna()

    features = [
        "temperatur_c",
        "spaending_v",
        "stroem_a",
        "fejl_antal",
        "oppetid_procent",
        "maks_effekt_kw",
        "charger_age_days",
    ]

    X = df[features]

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
    )

    model.fit(X)

    df["anomaly"] = model.predict(X)
    counts = df["anomaly"].value_counts().to_dict()

    print("\nModel 2 - Anomaly Detection")
    print("-" * 50)
    print("Normal   :", counts.get(1, 0))
    print("Anomaly  :", counts.get(-1, 0))

    model_report["Model 2 - Anomaly Detection"] = {
        "type": "anomaly_detection",
        "normal_count": int(counts.get(1, 0)),
        "anomaly_count": int(counts.get(-1, 0)),
        "contamination": 0.05,
        "features": features,
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "anomaly_detection_model.pkl"))

    anomalies = df[df["anomaly"] == -1]
    anomalies.to_csv(os.path.join(MODEL_DIR, "detected_anomalies.csv"), index=False)


def build_service_label(data):
    drift = add_charger_age(
        data["driftstilstande"],
        data["ladepunkter"]
    ).copy()

    service = data["servicehaendelser"].copy()

    drift["maaletidspunkt"] = pd.to_datetime(drift["maaletidspunkt"])
    service["service_dato"] = pd.to_datetime(service["service_dato"])

    drift["service_next_30_days"] = 0

    for index, row in drift.iterrows():
        same_charger = service[service["ladepunkt_id"] == row["ladepunkt_id"]]

        future_service = same_charger[
            (same_charger["service_dato"] > row["maaletidspunkt"])
            & (same_charger["service_dato"] <= row["maaletidspunkt"] + pd.Timedelta(days=30))
        ]

        if len(future_service) > 0:
            drift.at[index, "service_next_30_days"] = 1

    return drift


def train_service_prediction_model(data):
    df = build_service_label(data).dropna()

    features = [
        "driftsscore",
        "temperatur_c",
        "spaending_v",
        "stroem_a",
        "fejl_antal",
        "oppetid_procent",
        "maks_effekt_kw",
        "charger_age_days",
    ]

    target = "service_next_30_days"

    X = df[features]
    y = df[target]

    if y.nunique() < 2:
        print("\nModel 3 - Service Prediction skipped")
        print("-" * 50)
        print("Only one class found in target.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    name = "Model 3 - Service Prediction Next 30 Days"

    print(f"\n{name}")
    print("-" * 50)
    print("Accuracy:", round(accuracy, 3))
    print(classification_report(y_test, predictions, zero_division=0))

    model_report[name] = {
        "type": "classification",
        "accuracy": round(float(accuracy), 3),
        "positive_cases": int(y.sum()),
        "negative_cases": int((y == 0).sum()),
    }

    print_feature_importance(name, model, features)

    joblib.dump(model, os.path.join(MODEL_DIR, "service_prediction_model.pkl"))


def train_duration_model(data):
    sessions = data["opladninger"].copy()
    cars = data["elbiler"].copy()
    points = data["ladepunkter"].copy()

    df = sessions.merge(cars, on="elbil_id", how="left")
    df = df.merge(points, on="ladepunkt_id", how="left")
    df = df.dropna()

    df["starttid"] = pd.to_datetime(df["starttid"])
    df["hour"] = df["starttid"].dt.hour
    df["month"] = df["starttid"].dt.month
    df["weekday"] = df["starttid"].dt.weekday

    features = [
        "energi_kwh",
        "temperatur_c",
        "batterikapacitet_kwh",
        "maks_effekt_kw",
        "hour",
        "month",
        "weekday",
    ]

    target = "varighed_minutter"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    name = "Model 4 - Charging Duration Prediction"
    regression_metrics(name, y_test, predictions)
    print_feature_importance(name, model, features)

    joblib.dump(model, os.path.join(MODEL_DIR, "charging_duration_model.pkl"))


def train_daily_energy_demand_model(data):
    df = data["opladninger"].copy().dropna()

    df["starttid"] = pd.to_datetime(df["starttid"])
    df["date"] = df["starttid"].dt.date

    daily = df.groupby("date").agg(
        daily_energy=("energi_kwh", "sum"),
        session_count=("opladning_id", "count"),
        avg_temperature=("temperatur_c", "mean"),
        avg_price=("pris_per_kwh", "mean"),
        avg_duration=("varighed_minutter", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    daily["weekday"] = daily["date"].dt.weekday
    daily["day_of_year"] = daily["date"].dt.dayofyear

    daily = daily.sort_values("date")
    daily["previous_day_energy"] = daily["daily_energy"].shift(1)
    daily["previous_7_day_energy"] = daily["daily_energy"].rolling(7).mean().shift(1)

    daily = daily.dropna()

    features = [
        "session_count",
        "avg_temperature",
        "avg_price",
        "avg_duration",
        "month",
        "weekday",
        "day_of_year",
        "previous_day_energy",
        "previous_7_day_energy",
    ]

    target = "daily_energy"

    X = daily[features]
    y = daily[target]

    if len(daily) < 30:
        print("\nModel 5 - Daily Energy Demand skipped: not enough data")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    name = "Model 5 - Daily Energy Demand Forecast"
    regression_metrics(name, y_test, predictions)
    print_feature_importance(name, model, features)

    joblib.dump(model, os.path.join(MODEL_DIR, "daily_energy_demand_model.pkl"))


def train_revenue_forecast_model(data):
    df = data["opladninger"].copy().dropna()

    df["starttid"] = pd.to_datetime(df["starttid"])
    df["date"] = df["starttid"].dt.date

    daily = df.groupby("date").agg(
        daily_revenue=("samlet_pris", "sum"),
        daily_energy=("energi_kwh", "sum"),
        session_count=("opladning_id", "count"),
        avg_temperature=("temperatur_c", "mean"),
        avg_price=("pris_per_kwh", "mean"),
        avg_duration=("varighed_minutter", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    daily["weekday"] = daily["date"].dt.weekday
    daily["day_of_year"] = daily["date"].dt.dayofyear

    daily = daily.sort_values("date")
    daily["previous_day_revenue"] = daily["daily_revenue"].shift(1)
    daily["previous_7_day_revenue"] = daily["daily_revenue"].rolling(7).mean().shift(1)

    daily = daily.dropna()

    features = [
        "daily_energy",
        "session_count",
        "avg_temperature",
        "avg_price",
        "avg_duration",
        "month",
        "weekday",
        "day_of_year",
        "previous_day_revenue",
        "previous_7_day_revenue",
    ]

    target = "daily_revenue"

    X = daily[features]
    y = daily[target]

    if len(daily) < 30:
        print("\nModel 6 - Revenue Forecast skipped: not enough data")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    name = "Model 6 - Daily Revenue Forecast"
    regression_metrics(name, y_test, predictions)
    print_feature_importance(name, model, features)

    joblib.dump(model, os.path.join(MODEL_DIR, "daily_revenue_forecast_model.pkl"))


def train_customer_segmentation_model(data):
    sessions = data["opladninger"].copy().dropna()

    customer_data = sessions.groupby("kunde_id").agg(
        total_energy=("energi_kwh", "sum"),
        total_spend=("samlet_pris", "sum"),
        avg_duration=("varighed_minutter", "mean"),
        session_count=("opladning_id", "count"),
        failed_sessions=("status", lambda x: (x == "Failed").sum()),
        interrupted_sessions=("status", lambda x: (x == "Interrupted").sum()),
    ).reset_index()

    features = [
        "total_energy",
        "total_spend",
        "avg_duration",
        "session_count",
        "failed_sessions",
        "interrupted_sessions",
    ]

    X = customer_data[features]

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    customer_data["segment"] = model.fit_predict(X)

    print("\nModel 7 - Customer Segmentation")
    print("-" * 50)
    print(customer_data["segment"].value_counts())

    model_report["Model 7 - Customer Segmentation"] = {
        "type": "clustering",
        "clusters": customer_data["segment"].value_counts().to_dict(),
        "features": features,
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "customer_segmentation_model.pkl"))
    customer_data.to_csv(os.path.join(MODEL_DIR, "customer_segments.csv"), index=False)


def train_failure_risk_model(data):
    df = build_service_label(data).dropna()

    df["high_failure_risk"] = (
        (df["service_next_30_days"] == 1)
        | (df["driftsscore"] < 65)
        | (df["fejl_antal"] >= 3)
        | (df["oppetid_procent"] < 94)
    ).astype(int)

    features = [
        "driftsscore",
        "temperatur_c",
        "spaending_v",
        "stroem_a",
        "fejl_antal",
        "oppetid_procent",
        "maks_effekt_kw",
        "charger_age_days",
    ]

    target = "high_failure_risk"

    X = df[features]
    y = df[target]

    if y.nunique() < 2:
        print("\nModel 8 - Failure Risk skipped")
        print("-" * 50)
        print("Only one class found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    name = "Model 8 - Charger Failure Risk Classification"

    print(f"\n{name}")
    print("-" * 50)
    print("Accuracy:", round(accuracy, 3))
    print(classification_report(y_test, predictions, zero_division=0))

    model_report[name] = {
        "type": "classification",
        "accuracy": round(float(accuracy), 3),
        "positive_cases": int(y.sum()),
        "negative_cases": int((y == 0).sum()),
    }

    print_feature_importance(name, model, features)

    joblib.dump(model, os.path.join(MODEL_DIR, "failure_risk_model.pkl"))


def main():
    print("Training VoltEdge ML v2 models")
    print("=" * 50)

    data = load_data()

    train_health_score_model(data)
    train_anomaly_detection_model(data)
    train_service_prediction_model(data)
    train_duration_model(data)
    train_daily_energy_demand_model(data)
    train_revenue_forecast_model(data)
    train_customer_segmentation_model(data)
    train_failure_risk_model(data)

    save_report()

    print("\nAll ML v2 models trained.")
    print("Models saved in:", MODEL_DIR)
    print("Report saved as:", os.path.join(MODEL_DIR, "model_report.json"))


if __name__ == "__main__":
    main()
  from plot_models import create_all_plots

create_all_plots()
