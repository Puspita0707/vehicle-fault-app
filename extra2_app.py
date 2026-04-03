import os
from dotenv import load_dotenv
import psycopg2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Create FastAPI app
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "app": "Vehicle Predictive Maintenance API",
        "message": "Open index.html in your browser for the dashboard.",
        "endpoints": {
            "POST /predict": "Upload a CSV to analyze vehicle risk",
            "GET /history": "Get last 7 prediction records",
            "GET /health": "Check API and database status",
        },
    }

# -------------------------------
# Load trained artifacts
# -------------------------------
model = joblib.load("vehicle_risk_model.pkl")
model_features = joblib.load("model_features.pkl")

# -------------------------------
# PostgreSQL Connection
# -------------------------------
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if DB_PASSWORD:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host="localhost",
        port="5432"
    )
else:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host="localhost",
        port="5432"
    )

cursor = conn.cursor()

# -------------------------------
# Auto Create Table
# -------------------------------
def create_table_if_not_exists():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk FLOAT NOT NULL,
            status VARCHAR(20) NOT NULL,
            time_to_fault FLOAT,
            root_causes TEXT,
            recommended_components TEXT
        );
    """)
    conn.commit()

create_table_if_not_exists()

HORIZON = 15

# ---------------------------------------------------------
# UPDATED: Climate Intelligence Engine (Research-Based)
# ---------------------------------------------------------
def calculate_environmental_stress(df):
    """
    Virtual Sensing: Infers external climate impact based on sensor correlations.
    Applied Research Concepts:
    - Arrhenius Law: 1.5% penalty per degree C above baseline (Thermal Decay).
    - Peck Model: 0.5% penalty per % humidity above baseline (Oxidation Rate).
    """
    recent = df.tail(30)
    
    # 1. Inferring Ambient Heat (Virtual Sensor)
    avg_coolant = recent["Coolant Temp (°C)"].mean()
    inferred_temp = 25 + (max(0, avg_coolant - 85) * 0.5) 
    
    # 2. Inferring Humidity (Virtual Sensor)
    volts_std = recent["Battery Voltage (V)"].std()
    inferred_humidity = min(95, 40 + (volts_std * 100)) 

    # 3. Arrhenius Law Calculation (Baseline: 35°C)
    # Industry standard for accelerated aging in automotive electronics
    temp_penalty = max(0, (inferred_temp - 35) * 0.015) 
    
    # 4. Peck Model Calculation (Baseline: 80%)
    # Models the exponential acceleration of corrosion in humid environments
    hum_penalty = max(0, (inferred_humidity - 80) * 0.005)

    stress_multiplier = 1.0 + temp_penalty + hum_penalty
    
    warnings = []
    if inferred_temp > 35:
        warnings.append(f"Thermal Stress ({round(inferred_temp)}°C): Arrhenius decay active ({round(temp_penalty*100, 1)}% penalty).")
    
    if inferred_humidity > 80:
        warnings.append(f"Oxidation Risk ({round(inferred_humidity)}%): Peck model penalty ({round(hum_penalty*100, 1)}% penalty).")

    return stress_multiplier, inferred_temp, inferred_humidity, warnings

# -------------------------------
# Dynamic Risk Classification
# -------------------------------
def classify_risk_dynamic(risk, risk_series):
    arr = np.array(risk_series)
    p50 = np.percentile(arr, 50)
    p80 = np.percentile(arr, 80)

    if risk >= p80:
        return "CRITICAL"
    elif risk >= p50:
        return "WARNING"
    else:
        return "SAFE"

# -------------------------------
# Feature Engineering
# -------------------------------
def build_features(df):
    win = 10
    out = pd.DataFrame(index=df.index)
    out["MAP (kPa)_std"] = df["MAP (kPa)"].rolling(win).std()
    out["Coolant Temp (°C)_std"] = df["Coolant Temp (°C)"].rolling(win).std()
    out["Battery Voltage (V)_std"] = df["Battery Voltage (V)"].rolling(win).std()
    out["Fuel Rail Pressure (bar)_mean"] = df["Fuel Rail Pressure (bar)"].rolling(win).mean()
    return out

# -------------------------------
# NEW: Unsupervised Anomaly Detection
# -------------------------------
def detect_anomalies(df):
    recent = df.tail(30).select_dtypes(include=[np.number])
    z_scores = (recent - recent.mean()) / (recent.std() + 1e-6)
    anomaly_count = (np.abs(z_scores) > 2.5).sum().sum()
    return int(anomaly_count)

# -------------------------------
# Root Cause Detection
# -------------------------------
def detect_root_cause(df):
    recent = df.tail(30)
    causes = []
    if recent["Battery Voltage (V)"].std() > 0.4:
        causes.append("Battery / Charging System Issue")
    if recent["Coolant Temp (°C)"].std() > 5:
        causes.append("Coolant Temperature Fluctuation")
    if recent["MAP (kPa)"].std() > 8:
        causes.append("Manifold Pressure Instability (MAP)")
    if recent["Fuel Rail Pressure (bar)"].mean() < 30:
        causes.append("Fuel Supply / Injector Pressure Drop")
    return causes if causes else ["No dominant sensor anomaly detected"]

# -------------------------------
# Sensor Severity %
# -------------------------------
def compute_sensor_severity(df):
    recent = df.tail(30)
    scores = {
        "Battery Voltage (V)": recent["Battery Voltage (V)"].std() / 0.4,
        "Coolant Temp (°C)": recent["Coolant Temp (°C)"].std() / 5,
        "MAP (kPa)": recent["MAP (kPa)"].std() / 8,
        "Fuel Rail Pressure (bar)": max(0, (35 - recent["Fuel Rail Pressure (bar)"].mean()) / 10)
    }
    total = sum(scores.values())
    if total == 0:
        return {k: 0 for k in scores}
    return {k: round((v / total) * 100, 1) for k, v in scores.items()}

# -------------------------------
# Component Mapping
# -------------------------------
def map_components_with_confidence(root_causes, sensor_severity):
    component_map = {
        "Battery / Charging System Issue": {"Battery": 1.0, "Alternator": 0.8, "Voltage Regulator": 0.6},
        "Coolant Temperature Fluctuation": {"Radiator": 1.0, "Thermostat": 0.8, "Coolant Pump": 0.7},
        "Manifold Pressure Instability (MAP)": {"MAP Sensor": 1.0, "Intake Manifold": 0.7, "Vacuum Lines": 0.6},
        "Fuel Supply / Injector Pressure Drop": {"Fuel Pump": 1.0, "Fuel Injector": 0.8, "Fuel Filter": 0.6}
    }
    sensor_to_cause = {
        "Battery Voltage (V)": "Battery / Charging System Issue",
        "Coolant Temp (°C)": "Coolant Temperature Fluctuation",
        "MAP (kPa)": "Manifold Pressure Instability (MAP)",
        "Fuel Rail Pressure (bar)": "Fuel Supply / Injector Pressure Drop"
    }

    component_scores = {}
    for sensor, severity in sensor_severity.items():
        cause = sensor_to_cause.get(sensor)
        if cause in root_causes and cause in component_map:
            for component, weight in component_map[cause].items():
                score = severity * weight
                component_scores[component] = component_scores.get(component, 0) + score

    if not component_scores:
        return []

    total_score = sum(component_scores.values())
    for comp in component_scores:
        component_scores[comp] = round((component_scores[comp] / total_score) * 100, 1)
    return sorted(component_scores.items(), key=lambda x: x[1], reverse=True)

# -------------------------------
# SWOT Analysis (Climate Integrated)
# -------------------------------
def generate_swot(sensor_severity, component_confidence, risk_series, climate_warnings):
    strengths, weaknesses, opportunities, threats = [], [], [], []
    for sensor, value in sensor_severity.items():
        if value < 20: strengths.append(f"{sensor} stable ({value}%)")
        elif value > 40: weaknesses.append(f"{sensor} anomaly ({value}%)")
    
    # Climate Factors as Threats
    threats.extend(climate_warnings)
    
    if component_confidence:
        top = component_confidence[0]
        threats.append(f"{top[0]} failure risk ({top[1]}%)")
    
    return {
        "strengths": strengths if strengths else ["Electrical stability"],
        "weaknesses": weaknesses if weaknesses else ["None detected"],
        "opportunities": ["Climate-adaptive maintenance plan"],
        "threats": threats if threats else ["No immediate environmental threats"]
    }

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(contents), encoding="latin1")

        df.columns = df.columns.str.strip()
        X_raw = build_features(df).dropna()
        if len(X_raw) < 5: return {"status": "INSUFFICIENT_DATA"}

        X = X_raw[model_features]
        recent = X.tail(60)
        risk_series = model.predict(recent)
        
        # 1. Base AI Risk
        base_risk = float(risk_series[-1])
        
        # 2. Climate Intelligence Logic (UPDATED: Arrhenius & Peck models)
        stress_mult, amb_temp, humidity, climate_warnings = calculate_environmental_stress(df)
        
        # 3. Final Adjusted Risk (Base * Climate Stress)
        final_risk = min(1.0, base_risk * stress_mult)
        
        status = classify_risk_dynamic(final_risk, risk_series)
        time_to_fault = (1 - final_risk) * HORIZON

        if "Time (sec)" in df.columns:
            time_axis = df["Time (sec)"].tail(len(risk_series)).tolist()
        else:
            time_axis = list(range(len(risk_series)))

        sensor_severity = compute_sensor_severity(df)
        root_causes = detect_root_cause(df)
        anomalies = detect_anomalies(df) 
        component_confidence = map_components_with_confidence(root_causes, sensor_severity)
        
        # 4. Generate Integrated SWOT
        swot = generate_swot(sensor_severity, component_confidence, risk_series, climate_warnings)

        recommended_str = ", ".join([f"{comp} ({conf}%)" for comp, conf in component_confidence])

        # Database Log
        cursor.execute("""
            INSERT INTO predictions 
            (file_name, risk, status, time_to_fault, root_causes, recommended_components)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (file.filename, final_risk, status, time_to_fault, ", ".join(root_causes), recommended_str))
        conn.commit()

        return {
            "status": status,
            "risk": round(final_risk, 3),
            "base_risk": round(base_risk, 3),
            "climate_stress_pct": round((stress_mult - 1) * 100, 1),
            "ambient_temp": round(amb_temp, 1),
            "humidity": round(humidity, 1),
            "time_to_fault": round(time_to_fault, 2),
            "anomalies": anomalies,
            "risk_trend": [round(float(x), 3) for x in risk_series],
            "time_axis": time_axis,
            "root_causes": root_causes,
            "sensor_severity": sensor_severity,
            "recommended_components": component_confidence,
            "swot": swot
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history():
    cursor.execute("SELECT file_name, upload_time, risk, status, root_causes, recommended_components FROM predictions ORDER BY upload_time DESC LIMIT 7")
    rows = cursor.fetchall()
    return [{"file_name": r[0], "upload_time": r[1].strftime("%Y-%m-%d %H:%M"), "risk": float(r[2]), "status": r[3], "root_causes": r[4], "recommended_components": r[5]} for r in rows]

#new with climate intell eng