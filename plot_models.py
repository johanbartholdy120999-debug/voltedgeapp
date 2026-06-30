import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

DATABASE_URL = "sqlite:///./voltedge.db"
MODEL_DIR = "models_ml"
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL)


def save_plot(filename):
    path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_data():
    return {
        "opladninger": pd.read_sql("SELECT * FROM opladninger", engine),
        "driftstilstande": pd.read_sql("SELECT * FROM driftstilstande", engine),
        "ladepunkter": pd.read_sql("SELECT * FROM ladepunkter", engine),
    }


def plot_model_performance():
    path = os.path.join(MODEL_DIR, "model_report.json")

    if not os.path.exists(path):
        print("model_report.json not found")
        return

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    names = []
    scores = []

    for model_name, values in report.items():
        if values.get("type") == "regression" and "r2" in values:
            names.append(model_name.replace("Model ", "M"))
            scores.append(values["r2"])

        if values.get("type") == "classification" and "accuracy" in values:
            names.append(model_name.replace("Model ", "M"))
            scores.append(values["accuracy"])

    plt.figure(figsize=(12, 6))
    plt.barh(names, scores)
    plt.title("VoltEdge ML Model Performance")
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.grid(axis="x", alpha=0.3)

    save_plot("model_performance.png")


def plot_health_distribution(data):
    df = data["driftstilstande"]

    plt.figure(figsize=(10, 6))
    plt.hist(df["driftsscore"].dropna(), bins=25)
    plt.title("Charger Health Score Distribution")
    plt.xlabel("Health score")
    plt.ylabel("Number of measurements")
    plt.grid(axis="y", alpha=0.3)

    save_plot("health_score_distribution.png")


def plot_anomaly_distribution():
    path = os.path.join(MODEL_DIR, "detected_anomalies.csv")

    if not os.path.exists(path):
        print("detected_anomalies.csv not found")
        return

    anomalies = pd.read_csv(path)

    normal_count = 500 - len(anomalies)
    anomaly_count = len(anomalies)

    plt.figure(figsize=(7, 5))
    plt.bar(["Normal", "Anomaly"], [normal_count, anomaly_count])
    plt.title("Anomaly Detection Results")
    plt.ylabel("Records")
    plt.grid(axis="y", alpha=0.3)

    save_plot("anomaly_distribution.png")


def plot_customer_segments():
    path = os.path.join(MODEL_DIR, "customer_segments.csv")

    if not os.path.exists(path):
        print("customer_segments.csv not found")
        return

    df = pd.read_csv(path)
    counts = df["segment"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Customer Segments")
    plt.xlabel("Segment")
    plt.ylabel("Customers")
    plt.grid(axis="y", alpha=0.3)

    save_plot("customer_segments.png")


def plot_feature_importance(model_file, features, title, filename):
    path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(path):
        print(f"{model_file} not found")
        return

    model = joblib.load(path)

    if not hasattr(model, "feature_importances_"):
        print(f"{model_file} has no feature_importances_")
        return

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"], importance["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.3)

    save_plot(filename)


def plot_daily_energy_forecast(data):
    model_path = os.path.join(MODEL_DIR, "daily_energy_demand_model.pkl")

    if not os.path.exists(model_path):
        print("daily_energy_demand_model.pkl not found")
        return

    model = joblib.load(model_path)

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

    X = daily[features]
    y = daily["daily_energy"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictions = model.predict(X_test)

    result = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions
    }).head(40)

    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result["actual"], marker="o", label="Actual")
    plt.plot(result.index, result["predicted"], marker="o", label="Predicted")
    plt.title("Daily Energy Demand Forecast")
    plt.xlabel("Test sample")
    plt.ylabel("Energy kWh")
    plt.legend()
    plt.grid(alpha=0.3)

    save_plot("daily_energy_forecast.png")


def plot_daily_revenue_forecast(data):
    model_path = os.path.join(MODEL_DIR, "daily_revenue_forecast_model.pkl")

    if not os.path.exists(model_path):
        print("daily_revenue_forecast_model.pkl not found")
        return

    model = joblib.load(model_path)

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

    X = daily[features]
    y = daily["daily_revenue"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictions = model.predict(X_test)

    result = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions
    }).head(40)

    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result["actual"], marker="o", label="Actual")
    plt.plot(result.index, result["predicted"], marker="o", label="Predicted")
    plt.title("Daily Revenue Forecast")
    plt.xlabel("Test sample")
    plt.ylabel("Revenue DKK")
    plt.legend()
    plt.grid(alpha=0.3)

    save_plot("daily_revenue_forecast.png")


def plot_correlation_matrix(data):
    df = data["driftstilstande"][[
        "driftsscore",
        "temperatur_c",
        "spaending_v",
        "stroem_a",
        "fejl_antal",
        "oppetid_procent"
    ]].dropna()

    corr = df.corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Operational Data Correlation Matrix")

    save_plot("correlation_matrix.png")


def create_all_plots():
    print("Creating VoltEdge ML plots...")

    data = load_data()

    plot_model_performance()
    plot_health_distribution(data)
    plot_anomaly_distribution()
    plot_customer_segments()

    plot_feature_importance(
        "health_score_model.pkl",
        [
            "temperatur_c",
            "spaending_v",
            "stroem_a",
            "fejl_antal",
            "oppetid_procent",
            "maks_effekt_kw",
            "charger_age_days",
        ],
        "Health Model Feature Importance",
        "health_feature_importance.png"
    )

    plot_feature_importance(
        "failure_risk_model.pkl",
        [
            "driftsscore",
            "temperatur_c",
            "spaending_v",
            "stroem_a",
            "fejl_antal",
            "oppetid_procent",
            "maks_effekt_kw",
            "charger_age_days",
        ],
        "Failure Risk Feature Importance",
        "failure_feature_importance.png"
    )

    plot_daily_energy_forecast(data)
    plot_daily_revenue_forecast(data)
    plot_correlation_matrix(data)

    print("All plots created in the plots folder.")


if __name__ == "__main__":
    create_all_plots()