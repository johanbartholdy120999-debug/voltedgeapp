import jsonGet-Content dashboard.py -First 5
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///./voltedge.db"
MODEL_DIR = Path("models_ml")
PLOT_DIR = Path("plots")

st.set_page_config(
    page_title="VoltEdge Dashboard",
    page_icon="⚡",
    layout="wide",
)

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


@st.cache_data
def load_all_data():
    data = {
        "kunder": load_table("kunder"),
        "elbiler": load_table("elbiler"),
        "ladestationer": load_table("ladestationer"),
        "ladepunkter": load_table("ladepunkter"),
        "tariffer": load_table("tariffer"),
        "opladninger": load_table("opladninger"),
        "driftstilstande": load_table("driftstilstande"),
        "servicehaendelser": load_table("servicehaendelser"),
    }

    data["opladninger"]["starttid"] = pd.to_datetime(
        data["opladninger"]["starttid"],
        errors="coerce",
    )

    data["driftstilstande"]["maaletidspunkt"] = pd.to_datetime(
        data["driftstilstande"]["maaletidspunkt"],
        errors="coerce",
    )

    data["servicehaendelser"]["service_dato"] = pd.to_datetime(
        data["servicehaendelser"]["service_dato"],
        errors="coerce",
    )

    return data


@st.cache_resource
def load_model(filename: str):
    path = MODEL_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


def load_model_report():
    path = MODEL_DIR / "model_report.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


data = load_all_data()

health_model = load_model("health_score_model.pkl")
failure_model = load_model("failure_risk_model.pkl")
duration_model = load_model("charging_duration_model.pkl")
model_report = load_model_report()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def money(value):
    return f"{value:,.0f} DKK"


def number(value):
    return f"{value:,.0f}"


def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def show_saved_plot(filename: str):
    path = PLOT_DIR / filename
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Plot not found: {filename}. Run `python plot_models.py` first.")


def get_daily_charging():
    opladninger = data["opladninger"].copy()
    opladninger = opladninger.dropna(subset=["starttid"])

    opladninger["date"] = opladninger["starttid"].dt.date

    daily = opladninger.groupby("date").agg(
        energy_kwh=("energi_kwh", "sum"),
        revenue=("samlet_pris", "sum"),
        sessions=("opladning_id", "count"),
        avg_temperature=("temperatur_c", "mean"),
        avg_duration=("varighed_minutter", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("⚡ VoltEdge")
st.sidebar.caption("Predictive EV charging analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Charging Analytics",
        "Charger Health",
        "Machine Learning",
        "Predictions",
        "Plots Gallery",
        "Raw Data",
    ],
)

st.sidebar.divider()

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()


# --------------------------------------------------
# Overview page
# --------------------------------------------------

if page == "Overview":
    st.title("⚡ VoltEdge Dashboard")
    st.caption("Operational analytics, machine learning, and charging network insights.")

    opladninger = data["opladninger"]
    drift = data["driftstilstande"]

    total_customers = len(data["kunder"])
    total_sessions = len(opladninger)
    total_energy = opladninger["energi_kwh"].sum()
    total_revenue = opladninger["samlet_pris"].sum()
    avg_health = drift["driftsscore"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Customers", number(total_customers))
    col2.metric("Sessions", number(total_sessions))
    col3.metric("Energy", f"{total_energy:,.0f} kWh")
    col4.metric("Revenue", money(total_revenue))
    col5.metric("Avg health", f"{avg_health:.1f}%")

    st.divider()

    daily = get_daily_charging()

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.line(
            daily,
            x="date",
            y="revenue",
            title="Daily revenue",
            labels={"date": "Date", "revenue": "Revenue (DKK)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.line(
            daily,
            x="date",
            y="energy_kwh",
            title="Daily energy delivered",
            labels={"date": "Date", "energy_kwh": "Energy (kWh)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        membership_counts = data["kunder"]["medlemskabstype"].value_counts().reset_index()
        membership_counts.columns = ["membership", "count"]
        fig = px.pie(
            membership_counts,
            names="membership",
            values="count",
            title="Customer membership mix",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        station_sessions = (
            data["opladninger"]
            .merge(data["ladepunkter"], on="ladepunkt_id", how="left")
            .merge(data["ladestationer"], on="ladestation_id", how="left")
            .groupby("navn")
            .agg(sessions=("opladning_id", "count"))
            .reset_index()
            .sort_values("sessions", ascending=False)
        )

        fig = px.bar(
            station_sessions,
            x="sessions",
            y="navn",
            orientation="h",
            title="Charging sessions by station",
            labels={"sessions": "Sessions", "navn": "Station"},
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Charging Analytics
# --------------------------------------------------

elif page == "Charging Analytics":
    st.title("⚡ Charging Analytics")

    opladninger = data["opladninger"].copy()
    opladninger = opladninger.dropna(subset=["starttid"])
    opladninger["hour"] = opladninger["starttid"].dt.hour
    opladninger["weekday"] = opladninger["starttid"].dt.day_name()
    opladninger["month"] = opladninger["starttid"].dt.month_name()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            opladninger,
            x="energi_kwh",
            nbins=30,
            title="Charging energy distribution",
            labels={"energi_kwh": "Energy (kWh)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            opladninger,
            x="varighed_minutter",
            nbins=30,
            title="Charging duration distribution",
            labels={"varighed_minutter": "Duration (minutes)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        hour_data = opladninger.groupby("hour").size().reset_index(name="sessions")
        fig = px.bar(
            hour_data,
            x="hour",
            y="sessions",
            title="Sessions by hour of day",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            opladninger,
            x="energi_kwh",
            y="samlet_pris",
            size="varighed_minutter",
            hover_data=["temperatur_c", "pris_per_kwh", "status"],
            title="Energy delivered vs revenue",
            labels={
                "energi_kwh": "Energy (kWh)",
                "samlet_pris": "Revenue (DKK)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Charging duration vs energy")
    fig = px.scatter(
        opladninger,
        x="energi_kwh",
        y="varighed_minutter",
        color="status",
        hover_data=["temperatur_c", "pris_per_kwh"],
        title="Energy vs duration",
        labels={
            "energi_kwh": "Energy (kWh)",
            "varighed_minutter": "Duration (minutes)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Charger Health
# --------------------------------------------------

elif page == "Charger Health":
    st.title("❤️ Charger Health & Maintenance")

    drift = data["driftstilstande"].copy()
    service = data["servicehaendelser"].copy()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Measurements", number(len(drift)))
    col2.metric("Avg health", f"{drift['driftsscore'].mean():.1f}%")
    col3.metric("Avg uptime", f"{drift['oppetid_procent'].mean():.1f}%")
    col4.metric("Service events", number(len(service)))

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(
            drift,
            x="driftsscore",
            nbins=25,
            title="Health score distribution",
            labels={"driftsscore": "Health score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(
            drift,
            x="temperatur_c",
            y="driftsscore",
            size="fejl_antal",
            hover_data=["ladepunkt_id", "oppetid_procent", "stroem_a"],
            title="Temperature vs charger health",
            labels={
                "temperatur_c": "Temperature (°C)",
                "driftsscore": "Health score",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig = px.scatter(
            drift,
            x="fejl_antal",
            y="driftsscore",
            hover_data=["ladepunkt_id", "temperatur_c", "oppetid_procent"],
            title="Fault count vs health",
            labels={"fejl_antal": "Fault count", "driftsscore": "Health score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        fig = px.scatter(
            drift,
            x="spaending_v",
            y="stroem_a",
            color="fejl_antal",
            hover_data=["ladepunkt_id", "driftsscore"],
            title="Voltage vs current",
            labels={"spaending_v": "Voltage (V)", "stroem_a": "Current (A)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Service causes")
    cause_counts = service["aarsag"].value_counts().reset_index()
    cause_counts.columns = ["cause", "count"]

    fig = px.bar(
        cause_counts,
        x="count",
        y="cause",
        orientation="h",
        title="Service events by cause",
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Machine Learning
# --------------------------------------------------

elif page == "Machine Learning":
    st.title("🤖 Machine Learning Overview")

    if not model_report:
        st.warning("No model report found. Run `python train_model.py` first.")
    else:
        rows = []
        for name, values in model_report.items():
            if values.get("type") == "regression":
                rows.append({
                    "model": name,
                    "type": "Regression",
                    "score": values.get("r2"),
                    "metric": "R²",
                    "mae": values.get("mae"),
                    "rmse": values.get("rmse"),
                })
            elif values.get("type") == "classification":
                rows.append({
                    "model": name,
                    "type": "Classification",
                    "score": values.get("accuracy"),
                    "metric": "Accuracy",
                    "mae": None,
                    "rmse": None,
                })
            elif values.get("type") == "anomaly_detection":
                rows.append({
                    "model": name,
                    "type": "Anomaly",
                    "score": values.get("anomaly_count"),
                    "metric": "Anomalies",
                    "mae": None,
                    "rmse": None,
                })
            elif values.get("type") == "clustering":
                rows.append({
                    "model": name,
                    "type": "Clustering",
                    "score": len(values.get("clusters", {})),
                    "metric": "Clusters",
                    "mae": None,
                    "rmse": None,
                })

        report_df = pd.DataFrame(rows)
        st.dataframe(report_df, use_container_width=True)

        score_df = report_df[report_df["metric"].isin(["R²", "Accuracy"])].copy()

        fig = px.bar(
            score_df,
            x="score",
            y="model",
            color="type",
            orientation="h",
            title="Model performance",
            labels={"score": "Score", "model": "Model"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature importance")

        importance_rows = []
        for model_name, values in model_report.items():
            importance = values.get("feature_importance")
            if importance:
                for feature, score in importance.items():
                    importance_rows.append({
                        "model": model_name,
                        "feature": feature,
                        "importance": score,
                    })

        if importance_rows:
            importance_df = pd.DataFrame(importance_rows)

            selected_model = st.selectbox(
                "Select model",
                sorted(importance_df["model"].unique()),
            )

            filtered = importance_df[importance_df["model"] == selected_model].sort_values(
                "importance",
                ascending=True,
            )

            fig = px.bar(
                filtered,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Feature importance: {selected_model}",
            )
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

elif page == "Predictions":
    st.title("🔮 Live ML Predictions")

    tab1, tab2, tab3 = st.tabs(["Health score", "Failure risk", "Charging duration"])

    with tab1:
        st.subheader("Predict charger health")

        col1, col2, col3 = st.columns(3)

        temperature = col1.number_input("Temperature °C", value=28.0)
        voltage = col2.number_input("Voltage V", value=400.0)
        current = col3.number_input("Current A", value=120.0)

        col4, col5, col6, col7 = st.columns(4)

        faults = col4.number_input("Fault count", value=1, step=1)
        uptime = col5.number_input("Uptime %", value=98.5)
        power = col6.number_input("Charger power kW", value=150.0)
        age = col7.number_input("Charger age days", value=500, step=1)

        if st.button("Predict health"):
            if health_model is None:
                st.error("Health model not found. Run train_model.py first.")
            else:
                prediction = health_model.predict([[
                    temperature,
                    voltage,
                    current,
                    faults,
                    uptime,
                    power,
                    age,
                ]])[0]

                st.metric("Predicted health score", f"{prediction:.1f}%")

    with tab2:
        st.subheader("Predict failure risk")

        col1, col2, col3, col4 = st.columns(4)

        health = col1.number_input("Current health score", value=75.0)
        temperature = col2.number_input("Temperature °C ", value=35.0)
        voltage = col3.number_input("Voltage V ", value=400.0)
        current = col4.number_input("Current A ", value=180.0)

        col5, col6, col7, col8 = st.columns(4)

        faults = col5.number_input("Fault count ", value=2, step=1)
        uptime = col6.number_input("Uptime % ", value=96.0)
        power = col7.number_input("Power kW ", value=150.0)
        age = col8.number_input("Age days ", value=800, step=1)

        if st.button("Predict failure risk"):
            if failure_model is None:
                st.error("Failure risk model not found. Run train_model.py first.")
            else:
                features = [[
                    health,
                    temperature,
                    voltage,
                    current,
                    faults,
                    uptime,
                    power,
                    age,
                ]]

                prediction = failure_model.predict(features)[0]

                probability = None
                if hasattr(failure_model, "predict_proba"):
                    probability = failure_model.predict_proba(features)[0][1]

                risk = "HIGH" if int(prediction) == 1 else "LOW"

                st.metric("Failure risk", risk)

                if probability is not None:
                    st.metric("Failure probability", f"{probability:.0%}")

    with tab3:
        st.subheader("Predict charging duration")

        col1, col2, col3, col4 = st.columns(4)

        energy = col1.number_input("Energy kWh", value=45.0)
        temperature = col2.number_input("Temperature °C  ", value=12.0)
        battery = col3.number_input("Battery capacity kWh", value=82.0)
        power = col4.number_input("Charger power kW  ", value=150.0)

        col5, col6, col7 = st.columns(3)

        hour = col5.slider("Hour", 0, 23, 14)
        month = col6.slider("Month", 1, 12, 6)
        weekday = col7.slider("Weekday", 0, 6, 2)

        if st.button("Predict duration"):
            if duration_model is None:
                st.error("Duration model not found. Run train_model.py first.")
            else:
                prediction = duration_model.predict([[
                    energy,
                    temperature,
                    battery,
                    power,
                    hour,
                    month,
                    weekday,
                ]])[0]

                st.metric("Predicted duration", f"{prediction:.0f} minutes")


# --------------------------------------------------
# Plots Gallery
# --------------------------------------------------

elif page == "Plots Gallery":
    st.title("🖼️ Plot Gallery")

    if not PLOT_DIR.exists():
        st.warning("No plots folder found. Run `python plot_models.py` first.")
    else:
        files = sorted(PLOT_DIR.glob("*.png"))

        if not files:
            st.warning("No plot images found. Run `python plot_models.py` first.")
        else:
            cols = st.columns(2)

            for index, file in enumerate(files):
                with cols[index % 2]:
                    st.subheader(file.stem.replace("_", " ").title())
                    st.image(str(file), use_container_width=True)


# --------------------------------------------------
# Raw Data
# --------------------------------------------------

elif page == "Raw Data":
    st.title("📋 Raw Data Explorer")

    table_name = st.selectbox(
        "Select table",
        list(data.keys()),
    )

    df = data[table_name]
    st.write(f"Rows: {len(df)}")
    st.dataframe(df, use_container_width=True)
    
