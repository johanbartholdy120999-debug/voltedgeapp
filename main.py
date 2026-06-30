import json
import os
from pathlib import Path
from datetime import datetime, date, time

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import (
    Kunde,
    Elbil,
    Ladestation,
    Ladepunkt,
    Tarif,
    Opladning,
    Driftstilstand,
    Servicehaendelse,
)

app = FastAPI(title="VoltEdge API", version="1.0.0")

Base.metadata.create_all(bind=engine)

PLOT_DIR = Path("plots")
MODEL_DIR = Path("models_ml")

if PLOT_DIR.exists():
    app.mount("/static-plots", StaticFiles(directory=str(PLOT_DIR)), name="static-plots")


# --------------------------------------------------
# Database dependency
# --------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------
# ML model loading
# --------------------------------------------------

def load_model(filename: str):
    path = MODEL_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


health_model = load_model("health_score_model.pkl")
duration_model = load_model("charging_duration_model.pkl")
failure_model = load_model("failure_risk_model.pkl")


# --------------------------------------------------
# Schemas
# --------------------------------------------------

class KundeCreate(BaseModel):
    navn: str
    email: str
    telefon: str | None = None
    medlemskabstype: str | None = None


class ElbilCreate(BaseModel):
    kunde_id: int
    maerke: str | None = None
    model: str | None = None
    batterikapacitet_kwh: float | None = None
    produktionsaar: int | None = None


class LadestationCreate(BaseModel):
    navn: str
    by: str | None = None
    adresse: str | None = None
    breddegrad: float | None = None
    laengdegrad: float | None = None


class LadepunktCreate(BaseModel):
    ladestation_id: int
    ladetype: str | None = None
    maks_effekt_kw: float | None = None
    installationsdato: date | None = None
    status: str | None = None


class TarifCreate(BaseModel):
    ladestation_id: int
    navn: str | None = None
    tariftype: str | None = None
    pris_per_kwh: float | None = None
    start_tid: time | None = None
    slut_tid: time | None = None


class OpladningCreate(BaseModel):
    kunde_id: int
    elbil_id: int
    ladepunkt_id: int
    tarif_id: int
    energi_kwh: float
    varighed_minutter: int | None = None
    temperatur_c: float | None = None
    pris_per_kwh: float | None = None
    samlet_pris: float | None = None
    starttid: datetime | None = None
    sluttid: datetime | None = None
    status: str | None = "Completed"


class DriftstilstandCreate(BaseModel):
    ladepunkt_id: int
    driftsscore: float | None = None
    temperatur_c: float | None = None
    spaending_v: float | None = None
    stroem_a: float | None = None
    fejl_antal: int | None = None
    oppetid_procent: float | None = None
    maaletidspunkt: datetime | None = None


class ServicehaendelseCreate(BaseModel):
    ladepunkt_id: int
    servicetype: str | None = None
    aarsag: str | None = None
    service_dato: datetime | None = None
    nedetid_timer: float | None = None
    beskrivelse: str | None = None


class HealthPrediction(BaseModel):
    temperatur_c: float
    spaending_v: float
    stroem_a: float
    fejl_antal: int
    oppetid_procent: float
    maks_effekt_kw: float
    charger_age_days: int


class DurationPrediction(BaseModel):
    energi_kwh: float
    temperatur_c: float
    batterikapacitet_kwh: float
    maks_effekt_kw: float
    hour: int
    month: int
    weekday: int


class FailurePrediction(BaseModel):
    driftsscore: float
    temperatur_c: float
    spaending_v: float
    stroem_a: float
    fejl_antal: int
    oppetid_procent: float
    maks_effekt_kw: float
    charger_age_days: int


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def health():
    return {
        "status": "running",
        "service": "VoltEdge API",
        "docs": "/docs",
        "plots": "/plots",
        "plot_gallery": "/plots/gallery",
        "ml_report": "/ml/report",
    }


# --------------------------------------------------
# Generic helper
# --------------------------------------------------

def get_or_404(db: Session, model, object_id_name: str, object_id: int):
    item = db.query(model).filter(getattr(model, object_id_name) == object_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


# --------------------------------------------------
# Kunde endpoints
# --------------------------------------------------

@app.post("/kunder")
def create_kunde(data: KundeCreate, db: Session = Depends(get_db)):
    item = Kunde(**data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/kunder")
def get_kunder(db: Session = Depends(get_db)):
    return db.query(Kunde).all()


@app.get("/kunder/{kunde_id}")
def get_kunde(kunde_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Kunde, "kunde_id", kunde_id)


@app.delete("/kunder/{kunde_id}")
def delete_kunde(kunde_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Kunde, "kunde_id", kunde_id)
    db.delete(item)
    db.commit()
    return {"message": "Kunde deleted", "kunde_id": kunde_id}


# --------------------------------------------------
# Elbil endpoints
# --------------------------------------------------

@app.post("/elbiler")
def create_elbil(data: ElbilCreate, db: Session = Depends(get_db)):
    item = Elbil(**data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/elbiler")
def get_elbiler(db: Session = Depends(get_db)):
    return db.query(Elbil).all()


@app.get("/elbiler/{elbil_id}")
def get_elbil(elbil_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Elbil, "elbil_id", elbil_id)


@app.delete("/elbiler/{elbil_id}")
def delete_elbil(elbil_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Elbil, "elbil_id", elbil_id)
    db.delete(item)
    db.commit()
    return {"message": "Elbil deleted", "elbil_id": elbil_id}


# --------------------------------------------------
# Ladestation endpoints
# --------------------------------------------------

@app.post("/ladestationer")
def create_ladestation(data: LadestationCreate, db: Session = Depends(get_db)):
    item = Ladestation(**data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/ladestationer")
def get_ladestationer(db: Session = Depends(get_db)):
    return db.query(Ladestation).all()


@app.get("/ladestationer/{ladestation_id}")
def get_ladestation(ladestation_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Ladestation, "ladestation_id", ladestation_id)


@app.delete("/ladestationer/{ladestation_id}")
def delete_ladestation(ladestation_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Ladestation, "ladestation_id", ladestation_id)
    db.delete(item)
    db.commit()
    return {"message": "Ladestation deleted", "ladestation_id": ladestation_id}


# --------------------------------------------------
# Ladepunkt endpoints
# --------------------------------------------------

@app.post("/ladepunkter")
def create_ladepunkt(data: LadepunktCreate, db: Session = Depends(get_db)):
    payload = data.model_dump()
    if payload.get("installationsdato") is None:
        payload.pop("installationsdato", None)
    item = Ladepunkt(**payload)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/ladepunkter")
def get_ladepunkter(db: Session = Depends(get_db)):
    return db.query(Ladepunkt).all()


@app.get("/ladepunkter/{ladepunkt_id}")
def get_ladepunkt(ladepunkt_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Ladepunkt, "ladepunkt_id", ladepunkt_id)


@app.delete("/ladepunkter/{ladepunkt_id}")
def delete_ladepunkt(ladepunkt_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Ladepunkt, "ladepunkt_id", ladepunkt_id)
    db.delete(item)
    db.commit()
    return {"message": "Ladepunkt deleted", "ladepunkt_id": ladepunkt_id}


# --------------------------------------------------
# Tarif endpoints
# --------------------------------------------------

@app.post("/tariffer")
def create_tarif(data: TarifCreate, db: Session = Depends(get_db)):
    item = Tarif(**data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/tariffer")
def get_tariffer(db: Session = Depends(get_db)):
    return db.query(Tarif).all()


@app.get("/tariffer/{tarif_id}")
def get_tarif(tarif_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Tarif, "tarif_id", tarif_id)


@app.delete("/tariffer/{tarif_id}")
def delete_tarif(tarif_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Tarif, "tarif_id", tarif_id)
    db.delete(item)
    db.commit()
    return {"message": "Tarif deleted", "tarif_id": tarif_id}


# --------------------------------------------------
# Opladning endpoints
# --------------------------------------------------

@app.post("/opladninger")
def create_opladning(data: OpladningCreate, db: Session = Depends(get_db)):
    payload = data.model_dump()
    if payload.get("starttid") is None:
        payload["starttid"] = datetime.utcnow()
    item = Opladning(**payload)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/opladninger")
def get_opladninger(db: Session = Depends(get_db)):
    return db.query(Opladning).all()


@app.get("/opladninger/{opladning_id}")
def get_opladning(opladning_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Opladning, "opladning_id", opladning_id)


@app.delete("/opladninger/{opladning_id}")
def delete_opladning(opladning_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Opladning, "opladning_id", opladning_id)
    db.delete(item)
    db.commit()
    return {"message": "Opladning deleted", "opladning_id": opladning_id}


# --------------------------------------------------
# Driftstilstand endpoints
# --------------------------------------------------

@app.post("/driftstilstande")
def create_driftstilstand(data: DriftstilstandCreate, db: Session = Depends(get_db)):
    payload = data.model_dump()
    if payload.get("maaletidspunkt") is None:
        payload["maaletidspunkt"] = datetime.utcnow()
    item = Driftstilstand(**payload)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/driftstilstande")
def get_driftstilstande(db: Session = Depends(get_db)):
    return db.query(Driftstilstand).all()


@app.get("/driftstilstande/{driftstilstand_id}")
def get_driftstilstand(driftstilstand_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Driftstilstand, "driftstilstand_id", driftstilstand_id)


@app.delete("/driftstilstande/{driftstilstand_id}")
def delete_driftstilstand(driftstilstand_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Driftstilstand, "driftstilstand_id", driftstilstand_id)
    db.delete(item)
    db.commit()
    return {"message": "Driftstilstand deleted", "driftstilstand_id": driftstilstand_id}


# --------------------------------------------------
# Servicehaendelse endpoints
# --------------------------------------------------

@app.post("/servicehaendelser")
def create_servicehaendelse(data: ServicehaendelseCreate, db: Session = Depends(get_db)):
    payload = data.model_dump()
    if payload.get("service_dato") is None:
        payload["service_dato"] = datetime.utcnow()
    item = Servicehaendelse(**payload)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/servicehaendelser")
def get_servicehaendelser(db: Session = Depends(get_db)):
    return db.query(Servicehaendelse).all()


@app.get("/servicehaendelser/{servicehaendelse_id}")
def get_servicehaendelse(servicehaendelse_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Servicehaendelse, "servicehaendelse_id", servicehaendelse_id)


@app.delete("/servicehaendelser/{servicehaendelse_id}")
def delete_servicehaendelse(servicehaendelse_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, Servicehaendelse, "servicehaendelse_id", servicehaendelse_id)
    db.delete(item)
    db.commit()
    return {"message": "Servicehaendelse deleted", "servicehaendelse_id": servicehaendelse_id}


# --------------------------------------------------
# Analytics endpoints
# --------------------------------------------------

@app.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    total_energy = db.query(func.sum(Opladning.energi_kwh)).scalar() or 0
    total_revenue = db.query(func.sum(Opladning.samlet_pris)).scalar() or 0
    avg_health = db.query(func.avg(Driftstilstand.driftsscore)).scalar() or 0

    return {
        "customers": db.query(Kunde).count(),
        "vehicles": db.query(Elbil).count(),
        "stations": db.query(Ladestation).count(),
        "charging_points": db.query(Ladepunkt).count(),
        "charging_sessions": db.query(Opladning).count(),
        "service_events": db.query(Servicehaendelse).count(),
        "total_energy_kwh": round(float(total_energy), 2),
        "total_revenue_dkk": round(float(total_revenue), 2),
        "average_health_score": round(float(avg_health), 2),
    }


@app.get("/analytics/total-energy")
def total_energy(db: Session = Depends(get_db)):
    total = db.query(func.sum(Opladning.energi_kwh)).scalar() or 0
    return {"total_energy_kwh": round(float(total), 2)}


@app.get("/analytics/total-revenue")
def total_revenue(db: Session = Depends(get_db)):
    total = db.query(func.sum(Opladning.samlet_pris)).scalar() or 0
    return {"total_revenue_dkk": round(float(total), 2)}


# --------------------------------------------------
# ML prediction endpoints
# --------------------------------------------------

@app.post("/predict/health")
def predict_health(data: HealthPrediction):
    if health_model is None:
        raise HTTPException(status_code=404, detail="Health model not found. Run train_model.py first.")

    features = [[
        data.temperatur_c,
        data.spaending_v,
        data.stroem_a,
        data.fejl_antal,
        data.oppetid_procent,
        data.maks_effekt_kw,
        data.charger_age_days,
    ]]

    prediction = health_model.predict(features)[0]

    return {
        "predicted_health_score": round(float(prediction), 2)
    }


@app.post("/predict/duration")
def predict_duration(data: DurationPrediction):
    if duration_model is None:
        raise HTTPException(status_code=404, detail="Duration model not found. Run train_model.py first.")

    features = [[
        data.energi_kwh,
        data.temperatur_c,
        data.batterikapacitet_kwh,
        data.maks_effekt_kw,
        data.hour,
        data.month,
        data.weekday,
    ]]

    prediction = duration_model.predict(features)[0]

    return {
        "predicted_duration_minutes": round(float(prediction), 2)
    }


@app.post("/predict/failure-risk")
def predict_failure_risk(data: FailurePrediction):
    if failure_model is None:
        raise HTTPException(status_code=404, detail="Failure risk model not found. Run train_model.py first.")

    features = [[
        data.driftsscore,
        data.temperatur_c,
        data.spaending_v,
        data.stroem_a,
        data.fejl_antal,
        data.oppetid_procent,
        data.maks_effekt_kw,
        data.charger_age_days,
    ]]

    prediction = failure_model.predict(features)[0]

    probability = None
    if hasattr(failure_model, "predict_proba"):
        probability = float(failure_model.predict_proba(features)[0][1])

    return {
        "failure_risk": "HIGH" if int(prediction) == 1 else "LOW",
        "failure_probability": round(probability, 2) if probability is not None else None,
    }


@app.get("/ml/report")
def get_ml_report():
    path = MODEL_DIR / "model_report.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Model report not found. Run train_model.py first.")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/ml/customer-segments")
def get_customer_segments():
    path = MODEL_DIR / "customer_segments.csv"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Customer segments file not found. Run train_model.py first.")

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/ml/anomalies")
def get_anomalies():
    path = MODEL_DIR / "detected_anomalies.csv"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Anomalies file not found. Run train_model.py first.")

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


# --------------------------------------------------
# Plot endpoints
# --------------------------------------------------

@app.get("/plots")
def get_available_plots():
    if not PLOT_DIR.exists():
        return {
            "count": 0,
            "plots": [],
            "message": "No plots folder found. Run plot_models.py first.",
        }

    files = sorted(PLOT_DIR.glob("*.png"))

    return {
        "count": len(files),
        "plots": [
            {
                "name": file.stem,
                "filename": file.name,
                "url": f"/plots/{file.name}",
                "static_url": f"/static-plots/{file.name}",
            }
            for file in files
        ],
    }


@app.get("/plots/gallery", response_class=HTMLResponse)
def plot_gallery():
    if not PLOT_DIR.exists():
        return """
        <html>
            <body>
                <h1>No plots found</h1>
                <p>Run <code>python plot_models.py</code> first.</p>
            </body>
        </html>
        """

    images = sorted(PLOT_DIR.glob("*.png"))

    cards = ""

    for image in images:
        title = image.stem.replace("_", " ").title()
        cards += f"""
        <div class="card">
            <h2>{title}</h2>
            <img src="/plots/{image.name}" alt="{title}">
        </div>
        """

    return f"""
    <html>
        <head>
            <title>VoltEdge ML Plot Gallery</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f4f6f8;
                    margin: 0;
                    padding: 30px;
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
                    gap: 24px;
                }}
                .card {{
                    background: white;
                    border-radius: 14px;
                    padding: 20px;
                    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
                }}
                .card h2 {{
                    margin-top: 0;
                    font-size: 20px;
                }}
                img {{
                    width: 100%;
                    border-radius: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>VoltEdge Machine Learning Plot Gallery</h1>
            <div class="grid">
                {cards}
            </div>
        </body>
    </html>
    """


@app.get("/plots/{plot_name}")
def get_plot(plot_name: str):
    path = PLOT_DIR / plot_name

    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Plot not found")

    return FileResponse(path)
