#!/bin/bash

echo "=========================================="
echo "        VoltEdge Platform v1.0"
echo "=========================================="
echo ""

echo "Checking system..."

# Database check
if [ -f "voltedge.db" ]; then
    echo "✓ Database found"
else
    echo "✗ Database NOT found"
fi

# Models folder
if [ -d "models_ml" ]; then
    echo "✓ models_ml folder found"
else
    echo "✗ models_ml folder NOT found"
fi

# Health model
if [ -f "models_ml/health_score_model.pkl" ]; then
    echo "✓ Health model loaded"
else
    echo "✗ Health model missing"
fi

# Duration model
if [ -f "models_ml/charging_duration_model.pkl" ]; then
    echo "✓ Duration model loaded"
else
    echo "✗ Duration model missing"
fi

# Failure model
if [ -f "models_ml/failure_risk_model.pkl" ]; then
    echo "✓ Failure Risk model loaded"
else
    echo "✗ Failure Risk model missing"
fi

echo ""
echo "Starting FastAPI..."

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 &

sleep 5

echo "✓ FastAPI running"

echo ""
echo "Starting Streamlit Dashboard..."

streamlit run dashboard.py \
    --server.address=0.0.0.0 \
    --server.port=8501
