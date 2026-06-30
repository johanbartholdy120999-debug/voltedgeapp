import joblib

health_model = joblib.load("models_ml/health_score_model.pkl")

failure_model = joblib.load("models_ml/failure_risk_model.pkl")

duration_model = joblib.load("models_ml/charging_duration_model.pkl")

service_model = joblib.load("models_ml/service_prediction_model.pkl")