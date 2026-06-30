from pydantic import BaseModel


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