import os
import requests
from dotenv import load_dotenv
import psycopg2
from fastapi import FastAPI, UploadFile, File, Form # Added Form
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

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD if DB_PASSWORD else None,
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

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
# NEW: Visual Crossing Ground Truth Fetcher
# ---------------------------------------------------------
def get_visual_crossing_weather(city, timestamp_str):
    """
    Fetches actual historical weather for the specific diagnostic timeline.
    """
    api_key = os.getenv("VISUAL_CROSSING_KEY") # Ensure this is in your .env
    if not api_key or not city or not timestamp_str:
        return None, None
    
    # URL encoded city and timestamp (Supports ISO format from datetime-local)
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{timestamp_str}?unitGroup=metric&key={api_key}&include=current"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['currentConditions']['temp']
            hum = data['currentConditions']['humidity']
            return temp, hum
    except Exception as e:
        print(f"Weather API Fetch Failed: {e}")
    return None, None

# ---------------------------------------------------------
# UPDATED: Climate Intelligence Engine (Forensic Edition)
# ---------------------------------------------------------
def calculate_environmental_stress(df, actual_temp=None, actual_hum=None):
    """
    Combines Virtual Sensing with Internet Ground Truth.
    Uses Arrhenius Law (Thermal) and Peck Model (Oxidation).
    """
    recent = df.tail(30)
    
    # 1. Virtual Sensing (Fallback)
    avg_coolant = recent["Coolant Temp (°C)"].mean()
    v_temp = 25 + (max(0, avg_coolant - 85) * 0.5) 
    volts_std = recent["Battery Voltage (V)"].std()
    v_hum = min(95, 40 + (volts_std * 100)) 

    # 2. Selection: Use Internet Data if available, else Virtual
    final_temp = actual_temp if actual_temp is not None else v_temp
    final_hum = actual_hum if actual_hum is not None else v_hum

    # 3. Physics Models
    temp_penalty = max(0, (final_temp - 35) * 0.015) # Arrhenius
    hum_penalty = max(0, (final_hum - 80) * 0.005)   # Peck

    stress_multiplier = 1.0 + temp_penalty + hum_penalty
    
    source = "Verified Ground Truth" if actual_temp is not None else "Virtual Inference"
    warnings = []
    if final_temp > 35:
        warnings.append(f"Thermal Stress ({round(final_temp)}°C): Arrhenius penalty active via {source}.")
    if final_hum > 80:
        warnings.append(f"Oxidation Risk ({round(final_hum)}%): Peck penalty applied via {source}.")

    return stress_multiplier, final_temp, final_hum, warnings

# -------------------------------
# Helper Functions (Unchanged)
# -------------------------------
def classify_risk_dynamic(risk, risk_series):
    arr = np.array(risk_series)
    p50, p80 = np.percentile(arr, 50), np.percentile(arr, 80)
    if risk >= p80: return "CRITICAL"
    return "WARNING" if risk >= p50 else "SAFE"

def build_features(df):
    win = 10
    out = pd.DataFrame(index=df.index)
    out["MAP (kPa)_std"] = df["MAP (kPa)"].rolling(win).std()
    out["Coolant Temp (°C)_std"] = df["Coolant Temp (°C)"].rolling(win).std()
    out["Battery Voltage (V)_std"] = df["Battery Voltage (V)"].rolling(win).std()
    out["Fuel Rail Pressure (bar)_mean"] = df["Fuel Rail Pressure (bar)"].rolling(win).mean()
    return out

def detect_anomalies(df):
    recent = df.tail(30).select_dtypes(include=[np.number])
    z_scores = (recent - recent.mean()) / (recent.std() + 1e-6)
    return int((np.abs(z_scores) > 2.5).sum().sum())

def detect_root_cause(df):
    recent, causes = df.tail(30), []
    if recent["Battery Voltage (V)"].std() > 0.4: causes.append("Battery / Charging System Issue")
    if recent["Coolant Temp (°C)"].std() > 5: causes.append("Coolant Temperature Fluctuation")
    if recent["MAP (kPa)"].std() > 8: causes.append("Manifold Pressure Instability (MAP)")
    if recent["Fuel Rail Pressure (bar)"].mean() < 30: causes.append("Fuel Supply / Injector Pressure Drop")
    return causes if causes else ["No dominant sensor anomaly detected"]

def compute_sensor_severity(df):
    recent = df.tail(30)
    scores = {
        "Battery Voltage (V)": recent["Battery Voltage (V)"].std() / 0.4,
        "Coolant Temp (°C)": recent["Coolant Temp (°C)"].std() / 5,
        "MAP (kPa)": recent["MAP (kPa)"].std() / 8,
        "Fuel Rail Pressure (bar)": max(0, (35 - recent["Fuel Rail Pressure (bar)"].mean()) / 10)
    }
    total = sum(scores.values())
    return {k: round((v / total) * 100, 1) if total > 0 else 0 for k, v in scores.items()}

def map_components_with_confidence(root_causes, sensor_severity):
    component_map = {
        "Battery / Charging System Issue": {"Battery": 1.0, "Alternator": 0.8, "Voltage Regulator": 0.6},
        "Coolant Temperature Fluctuation": {"Radiator": 1.0, "Thermostat": 0.8, "Coolant Pump": 0.7},
        "Manifold Pressure Instability (MAP)": {"MAP Sensor": 1.0, "Intake Manifold": 0.7, "Vacuum Lines": 0.6},
        "Fuel Supply / Injector Pressure Drop": {"Fuel Pump": 1.0, "Fuel Injector": 0.8, "Fuel Filter": 0.6}
    }
    sensor_to_cause = {
        "Battery Voltage (V)": "Battery / Charging System Issue", "Coolant Temp (°C)": "Coolant Temperature Fluctuation",
        "MAP (kPa)": "Manifold Pressure Instability (MAP)", "Fuel Rail Pressure (bar)": "Fuel Supply / Injector Pressure Drop"
    }
    component_scores = {}
    for sensor, severity in sensor_severity.items():
        cause = sensor_to_cause.get(sensor)
        if cause in root_causes and cause in component_map:
            for component, weight in component_map[cause].items():
                component_scores[component] = component_scores.get(component, 0) + (severity * weight)
    if not component_scores: return []
    total = sum(component_scores.values())
    return sorted([(c, round((v/total)*100, 1)) for c, v in component_scores.items()], key=lambda x: x[1], reverse=True)

def generate_swot(sensor_severity, component_confidence, climate_warnings):
    s, w, o, t = [], [], ["Climate-adaptive maintenance plan"], []
    for sensor, val in sensor_severity.items():
        if val < 20: s.append(f"{sensor} stable ({val}%)")
        elif val > 40: w.append(f"{sensor} anomaly ({val}%)")
    t.extend(climate_warnings)
    if component_confidence: t.append(f"{component_confidence[0][0]} failure risk ({component_confidence[0][1]}%)")
    return {"strengths": s or ["Electrical stability"], "weaknesses": w or ["None detected"], "opportunities": o, "threats": t or ["No immediate threats"]}

# -------------------------------
# UPDATED Prediction Endpoint
# -------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    city: str = Form(None),      # Added Form support
    timestamp: str = Form(None)  # Added Form support
):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        X_raw = build_features(df).dropna()
        if len(X_raw) < 5: return {"status": "INSUFFICIENT_DATA"}

        risk_series = model.predict(X_raw[model_features])
        base_risk = float(risk_series[-1])
        
        # 1. Forensic Ground Truth Lookup
        actual_temp, actual_hum = get_visual_crossing_weather(city, timestamp)
        
        # 2. Climate Engine (Research-Based)
        stress_mult, final_t, final_h, warnings = calculate_environmental_stress(df, actual_temp, actual_hum)
        
        final_risk = min(1.0, base_risk * stress_mult)
        status = classify_risk_dynamic(final_risk, risk_series)
        ttf = (1 - final_risk) * HORIZON

        # Analysis Components
        severity = compute_sensor_severity(df)
        causes = detect_root_cause(df)
        comp_conf = map_components_with_confidence(causes, severity)
        swot = generate_swot(severity, comp_conf, warnings)

        # Database Log
        cursor.execute("""
            INSERT INTO predictions (file_name, risk, status, time_to_fault, root_causes, recommended_components)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (file.filename, final_risk, status, ttf, ", ".join(causes), ", ".join([f"{c}({p}%)" for c,p in comp_conf])))
        conn.commit()

        return {
            "status": status, "risk": round(final_risk, 3), "base_risk": round(base_risk, 3),
            "climate_stress_pct": round((stress_mult - 1) * 100, 1),
            "ambient_temp": round(final_t, 1), "humidity": round(final_h, 1),
            "time_to_fault": round(ttf, 2), "anomalies": detect_anomalies(df),
            "risk_trend": [round(float(x), 3) for x in risk_series],
            "time_axis": list(range(len(risk_series))),
            "root_causes": causes, "sensor_severity": severity, "recommended_components": comp_conf, "swot": swot
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history():
    cursor.execute("SELECT file_name, upload_time, risk, status, root_causes, recommended_components FROM predictions ORDER BY upload_time DESC LIMIT 7")
    return [{"file_name": r[0], "upload_time": r[1].strftime("%Y-%m-%d %H:%M"), "risk": float(r[2]), "status": r[3], "root_causes": r[4], "recommended_components": r[5]} for r in cursor.fetchall()]