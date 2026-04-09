import os
import json
import math
import asyncio
import random
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import psycopg2
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import joblib
import numpy as np
from io import BytesIO
from scipy.stats import linregress
from typing import List
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import bcrypt as bcrypt_lib
from jose import JWTError, jwt
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# ----------------------------------------------------------------
# JSON safety: replace NaN / ±Infinity with None so FastAPI never
# hits "Out of range float values are not JSON compliant"
# ----------------------------------------------------------------
def sanitize_floats(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_floats(v) for v in obj]
    return obj

# ================================================================
# AUTH CONFIGURATION
# ================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "vehicle-maintenance-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
security = HTTPBearer(auto_error=False)

# ================================================================
# PYDANTIC MODELS
# ================================================================
class UserRegister(BaseModel):
    username: str
    password: str
    email: str

class UserLogin(BaseModel):
    username: str
    password: str

class VehicleCreate(BaseModel):
    name: str
    vehicle_type: str  # "petrol_hatchback", "petrol_sedan", "diesel_suv", "diesel_truck"
    year: int = None
    make: str = None
    model_name: str = None

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
def serve_frontend():
    return FileResponse("index.html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })

# -------------------------------
# Load trained artifacts
# -------------------------------
model = joblib.load("vehicle_risk_model.pkl")
model_features = joblib.load("model_features.pkl")

# -------------------------------
# PostgreSQL Connection
# -------------------------------
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD if DB_PASSWORD else None,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.autocommit = True
    cursor = conn.cursor()
except Exception as e:
    print(f"WARNING: Could not connect to database at startup: {e}")
    conn = None
    cursor = None

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
    # Add new columns to existing table if they don't exist yet
    cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS user_id INTEGER;")
    cursor.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS vehicle_id INTEGER;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(100) NOT NULL,
            vehicle_type VARCHAR(50) NOT NULL DEFAULT 'petrol_hatchback',
            year INTEGER,
            make VARCHAR(50),
            model_name VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            vehicle_id INTEGER NOT NULL,
            component VARCHAR(100) NOT NULL,
            service_date DATE NOT NULL,
            notes TEXT,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

create_table_if_not_exists()

# ================================================================
# AUTH HELPERS
# ================================================================
def verify_password(plain_password, hashed_password):
    return bcrypt_lib.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def get_password_hash(password):
    return bcrypt_lib.hashpw(password.encode("utf-8"), bcrypt_lib.gensalt()).decode("utf-8")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None

HORIZON = 15

# ================================================================
# VEHICLE TYPE PROFILES
# Different vehicle types have different healthy sensor ranges
# ================================================================
VEHICLE_PROFILES = {
    "petrol_hatchback": {
        "rpm_redline": 6500,
        "rpm_normal_max": 4000,
        "coolant_normal_max": 95,
        "map_overload_threshold": 90,
        "battery_min": 12.0,
        "fuel_pressure_min_bar": 2.5,
        "label": "Petrol Hatchback"
    },
    "petrol_sedan": {
        "rpm_redline": 6000,
        "rpm_normal_max": 4000,
        "coolant_normal_max": 100,
        "map_overload_threshold": 92,
        "battery_min": 12.0,
        "fuel_pressure_min_bar": 2.8,
        "label": "Petrol Sedan"
    },
    "diesel_suv": {
        "rpm_redline": 4500,
        "rpm_normal_max": 3000,
        "coolant_normal_max": 98,
        "map_overload_threshold": 120,
        "battery_min": 11.5,
        "fuel_pressure_min_bar": 15.0,
        "label": "Diesel SUV"
    },
    "diesel_truck": {
        "rpm_redline": 3500,
        "rpm_normal_max": 2500,
        "coolant_normal_max": 100,
        "map_overload_threshold": 150,
        "battery_min": 11.0,
        "fuel_pressure_min_bar": 20.0,
        "label": "Diesel Truck"
    }
}

def get_vehicle_profile(vehicle_type: str = None):
    return VEHICLE_PROFILES.get(vehicle_type or "petrol_hatchback", VEHICLE_PROFILES["petrol_hatchback"])

# ================================================================
# EXISTING: Visual Crossing Weather (unchanged)
# ================================================================
def get_visual_crossing_weather(city, timestamp_str):
    api_key = os.getenv("VISUAL_CROSSING_KEY")
    if not api_key or not city or not timestamp_str:
        return None, None
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{timestamp_str}?unitGroup=metric&key={api_key}&include=current"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['currentConditions']['temp'], data['currentConditions']['humidity']
    except Exception as e:
        print(f"Weather API Fetch Failed: {e}")
    return None, None

# ================================================================
# EXISTING: Climate Engine (unchanged)
# ================================================================
def calculate_environmental_stress(df, actual_temp=None, actual_hum=None):
    recent = df.tail(30)
    # Safely handle CSVs that don't have these specific sensor columns
    raw_coolant = float(recent["Coolant Temp (°C)"].mean()) if "Coolant Temp (°C)" in df.columns else 80.0
    # Clamp to a realistic engine coolant range (0–150°C) — guards against raw ADC values
    # accidentally aliased to this column causing absurd temperature calculations
    avg_coolant = max(0.0, min(150.0, raw_coolant))
    v_temp = 25 + (max(0, avg_coolant - 85) * 0.5)
    volts_std = float(recent["Battery Voltage (V)"].std()) if "Battery Voltage (V)" in df.columns else 0.1
    v_hum = min(95, 40 + (volts_std * 100))
    final_temp = actual_temp if actual_temp is not None else float(v_temp)
    final_hum = actual_hum if actual_hum is not None else float(v_hum)
    temp_penalty = max(0, (final_temp - 35) * 0.015)
    hum_penalty = max(0, (final_hum - 80) * 0.005)
    stress_multiplier = 1.0 + temp_penalty + hum_penalty
    source = "Verified Ground Truth" if actual_temp is not None else "Virtual Inference"
    warnings = []
    if final_temp > 35:
        warnings.append(f"Thermal Stress ({round(final_temp)}°C): Arrhenius penalty active via {source}.")
    if final_hum > 80:
        warnings.append(f"Oxidation Risk ({round(final_hum)}%): Peck penalty applied via {source}.")
    return stress_multiplier, final_temp, final_hum, warnings

# ================================================================
# EXISTING: Climate Impact functions (unchanged)
# ================================================================
def quantify_climate_impact(base_risk, climate_adjusted_risk):
    if base_risk == 0:
        return 0.0
    return round(((climate_adjusted_risk - base_risk) / base_risk) * 100, 2)

EXTREME_PROFILES = {
    "desert_heat": {
        "label": "Desert Heat",
        "temp_threshold": 42,
        "arrhenius_multiplier": 2.8,
        "peck_multiplier": 0.6,
        "component_impact": {
            "Coolant System": {"risk_increase_pct": 85, "reason": "Coolant boils/evaporates faster above 42°C"},
            "Battery":        {"risk_increase_pct": 75, "reason": "Electrolyte evaporation & plate corrosion in extreme heat"},
            "Fuel System":    {"risk_increase_pct": 60, "reason": "Fuel vaporization and injector stress"},
            "MAP Sensor":     {"risk_increase_pct": 40, "reason": "Air density drops → MAP reads lower, incorrect fueling"},
        },
        "internal_sensor_shift": {
            "Coolant Temp (°C)":        "+12 to +20°C above baseline",
            "Battery Voltage (V)":      "-0.5 to -1.2V drop under thermal load",
            "Fuel Rail Pressure (bar)": "-8 to -15 bar (vapor lock risk)",
            "MAP (kPa)":                "-5 to -10 kPa (thin hot air)",
        }
    },
    "cold_snow": {
        "label": "Cold / Snow",
        "temp_threshold": -5,
        "arrhenius_multiplier": 0.4,
        "peck_multiplier": 1.9,
        "component_impact": {
            "Battery":        {"risk_increase_pct": 90, "reason": "Cold reduces chemical reaction rate; up to 50% capacity loss at -18°C"},
            "Fuel System":    {"risk_increase_pct": 70, "reason": "Fuel viscosity increases; injector clogging risk in diesel"},
            "Coolant System": {"risk_increase_pct": 55, "reason": "Freezing risk if antifreeze ratio insufficient"},
            "MAP Sensor":     {"risk_increase_pct": 30, "reason": "Dense cold air skews pressure readings"},
        },
        "internal_sensor_shift": {
            "Coolant Temp (°C)":        "-5 to -15°C slower warm-up, cold start stress",
            "Battery Voltage (V)":      "-0.8 to -2.0V (significant capacity loss)",
            "Fuel Rail Pressure (bar)": "+5 to +10 bar (viscous cold fuel)",
            "MAP (kPa)":                "+8 to +15 kPa (dense cold air)",
        }
    },
    "normal": {
        "label": "Normal / Optimal",
        "temp_threshold": None,
        "arrhenius_multiplier": 1.0,
        "peck_multiplier": 1.0,
        "component_impact": {
            "Coolant System": {"risk_increase_pct": 5,  "reason": "Minimal thermal stress"},
            "Battery":        {"risk_increase_pct": 5,  "reason": "Optimal operating range"},
            "Fuel System":    {"risk_increase_pct": 5,  "reason": "Normal fuel behavior"},
            "MAP Sensor":     {"risk_increase_pct": 3,  "reason": "Standard air density"},
        },
        "internal_sensor_shift": {
            "Coolant Temp (°C)":        "±2°C (baseline stable)",
            "Battery Voltage (V)":      "±0.1V (stable)",
            "Fuel Rail Pressure (bar)": "±2 bar (normal)",
            "MAP (kPa)":                "±3 kPa (normal)",
        }
    }
}

def classify_extreme_condition(temp, humidity):
    if temp >= 42: return "desert_heat"
    elif temp <= -5: return "cold_snow"
    return "normal"

def get_extreme_condition_analysis(temp, humidity, base_risk):
    condition_key = classify_extreme_condition(temp, humidity)
    profile = EXTREME_PROFILES[condition_key]
    combined_multiplier = (profile["arrhenius_multiplier"] * 0.6 + profile["peck_multiplier"] * 0.4)
    adjusted_risk = min(1.0, base_risk * combined_multiplier)
    return {
        "condition": condition_key,
        "condition_label": profile["label"],
        "combined_stress_multiplier": round(combined_multiplier, 3),
        "arrhenius_multiplier": profile["arrhenius_multiplier"],
        "peck_multiplier": profile["peck_multiplier"],
        "overall_risk_increase_pct": round((combined_multiplier - 1) * 100, 1),
        "adjusted_risk_under_condition": round(adjusted_risk, 4),
        "component_impact": profile["component_impact"],
        "internal_sensor_shift": profile["internal_sensor_shift"],
    }

def compute_climate_vs_internal_split(base_risk, final_risk, sensor_severity):
    climate_delta = final_risk - base_risk
    total_risk = final_risk if final_risk > 0 else 1e-6
    climate_share_pct = round((climate_delta / total_risk) * 100, 1)
    internal_share_pct = round(100 - climate_share_pct, 1)
    sensor_climate_sensitivity = {}
    for sensor, severity in sensor_severity.items():
        sensitivity = round(severity * (climate_delta / (base_risk + 1e-6)) * 100, 1)
        sensor_climate_sensitivity[sensor] = max(0, sensitivity)
    return {
        "climate_share_pct": max(0, climate_share_pct),
        "internal_share_pct": max(0, internal_share_pct),
        "sensor_climate_sensitivity": sensor_climate_sensitivity,
    }

# ================================================================
# DYNAMIC COLUMN NORMALIZATION
# Maps real-world CSV column name variants → internal standard names
# so the dashboard works with ANY vehicle's exported data
# ================================================================
COLUMN_ALIASES = {
    # Throttle / Accelerator
    "Accelerator Pedal Position (%)":        "Throttle Position (%)",
    "Accelerator Position (%)":              "Throttle Position (%)",
    "Pedal Position (%)":                    "Throttle Position (%)",
    "Throttle Pos (%)":                      "Throttle Position (%)",
    # Coolant temperature — NOTE: "Coolant Temperature Raw" intentionally excluded
    # because Raw columns store unprocessed ADC values, not real °C
    "Coolant Temperature (°C)":              "Coolant Temp (°C)",
    "Engine Coolant Temperature":            "Coolant Temp (°C)",
    "ECT (°C)":                              "Coolant Temp (°C)",
    "Coolant Temp":                          "Coolant Temp (°C)",
    # Engine RPM
    "RPM":                                   "Engine RPM",
    "Engine Speed (RPM)":                    "Engine RPM",
    "Engine Speed":                          "Engine RPM",
    "Crankshaft Speed (RPM)":               "Engine RPM",
    "Motor Speed (RPM)":                     "Engine RPM",
    # Vehicle speed
    "Speed (km/h)":                          "Vehicle Speed (km/h)",
    "Vehicle Speed":                         "Vehicle Speed (km/h)",
    "GPS Speed (km/h)":                      "Vehicle Speed (km/h)",
    "Speed":                                 "Vehicle Speed (km/h)",
    # MAP / Boost pressure (approximate mapping)
    "Boost Pressure (mV)":                   "MAP (kPa)",
    "Manifold Pressure (kPa)":              "MAP (kPa)",
    "Intake Manifold Pressure (kPa)":       "MAP (kPa)",
    "MAP":                                   "MAP (kPa)",
    "Boost Pressure (kPa)":                 "MAP (kPa)",
    # Battery / voltage
    "Battery Voltage":                       "Battery Voltage (V)",
    "Voltage (V)":                           "Battery Voltage (V)",
    "System Voltage (V)":                    "Battery Voltage (V)",
    # Fuel pressure
    "Fuel Pressure (kPa)":                   "Fuel Rail Pressure (bar)",
    "Fuel Rail Pressure (kPa)":             "Fuel Rail Pressure (bar)",
    "Fuel Pressure":                         "Fuel Rail Pressure (bar)",
    # Timestamp
    "Timestamp":                             "Timestamp (s)",
    "Time (s)":                              "Timestamp (s)",
    "Time":                                  "Timestamp (s)",
}

def normalize_columns(df):
    """
    Rename incoming CSV columns to internal standard names where aliases match.
    Un-matched columns are left as-is — they'll still appear in the sensor dropdown.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            target = COLUMN_ALIASES[col]
            if target not in df.columns:   # don't overwrite if target already exists
                rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# ================================================================
# EXISTING: Core helpers (unchanged)
# ================================================================
def classify_risk_dynamic(risk, risk_series):
    arr = np.array(risk_series)
    p50, p80 = np.percentile(arr, 50), np.percentile(arr, 80)
    if risk >= p80: return "CRITICAL"
    return "WARNING" if risk >= p50 else "SAFE"

def build_features(df):
    win = 10
    out = pd.DataFrame(index=df.index)
    # Only build features for columns that exist — returns NaN columns for missing ones
    # (dropna() in the caller will then exclude those rows, returning empty for new CSVs)
    out["MAP (kPa)_std"] = df["MAP (kPa)"].rolling(win).std() if "MAP (kPa)" in df.columns else np.nan
    out["Coolant Temp (°C)_std"] = df["Coolant Temp (°C)"].rolling(win).std() if "Coolant Temp (°C)" in df.columns else np.nan
    out["Battery Voltage (V)_std"] = df["Battery Voltage (V)"].rolling(win).std() if "Battery Voltage (V)" in df.columns else np.nan
    out["Fuel Rail Pressure (bar)_mean"] = df["Fuel Rail Pressure (bar)"].rolling(win).mean() if "Fuel Rail Pressure (bar)" in df.columns else np.nan
    return out

def detect_anomalies(df):
    recent = df.tail(30).select_dtypes(include=[np.number])
    z_scores = (recent - recent.mean()) / (recent.std() + 1e-6)
    return int((np.abs(z_scores) > 2.5).sum().sum())

def detect_root_cause(df):
    recent, causes = df.tail(30), []
    if "Battery Voltage (V)" in df.columns and recent["Battery Voltage (V)"].std() > 0.4:
        causes.append("Battery / Charging System Issue")
    if "Coolant Temp (°C)" in df.columns and recent["Coolant Temp (°C)"].std() > 5:
        causes.append("Coolant Temperature Fluctuation")
    if "MAP (kPa)" in df.columns and recent["MAP (kPa)"].std() > 8:
        causes.append("Manifold Pressure Instability (MAP)")
    if "Fuel Rail Pressure (bar)" in df.columns and recent["Fuel Rail Pressure (bar)"].mean() < 2.5:
        causes.append("Fuel Supply / Injector Pressure Drop")
    return causes if causes else ["No dominant sensor anomaly detected"]

def compute_sensor_severity(df):
    recent = df.tail(30)
    scores = {}
    # --- Known sensors with calibrated thresholds ---
    if "Battery Voltage (V)" in df.columns:
        scores["Battery Voltage (V)"] = recent["Battery Voltage (V)"].std() / 0.4
    if "Coolant Temp (°C)" in df.columns:
        scores["Coolant Temp (°C)"] = recent["Coolant Temp (°C)"].std() / 5
    if "MAP (kPa)" in df.columns:
        scores["MAP (kPa)"] = recent["MAP (kPa)"].std() / 8
    if "Fuel Rail Pressure (bar)" in df.columns:
        scores["Fuel Rail Pressure (bar)"] = max(0, (2.5 - recent["Fuel Rail Pressure (bar)"].mean()) / 0.5)
    if "Engine RPM" in df.columns:
        scores["Engine RPM"] = min(1.0, recent["Engine RPM"].std() / 500)
    if "Vehicle Speed (km/h)" in df.columns:
        scores["Vehicle Speed (km/h)"] = min(1.0, recent["Vehicle Speed (km/h)"].std() / 30)
    if "Throttle Position (%)" in df.columns:
        scores["Throttle Position (%)"] = min(1.0, recent["Throttle Position (%)"].std() / 20)
    # --- Generic fallback: coefficient of variation for any unrecognised numeric column ---
    # This makes the dashboard work with ANY vehicle's CSV regardless of column names
    exclude_cols = {"Timestamp (s)", "timestamp", "index", "row_id", "Unnamed: 0"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in scores and col not in exclude_cols:
            col_mean = abs(recent[col].mean()) + 1e-6
            col_std  = recent[col].std()
            # Coefficient of variation capped at 1.0 — works universally
            scores[col] = min(1.0, col_std / col_mean)
    total = sum(scores.values())
    return {k: round((v / total) * 100, 1) if total > 0 else 0 for k, v in scores.items()}

def map_components_with_confidence(root_causes, sensor_severity):
    # --- Path 1: ML root-cause based mapping (standard CSV with known columns) ---
    cause_component_map = {
        "Battery / Charging System Issue":      {"Battery": 1.0, "Alternator": 0.8, "Voltage Regulator": 0.6},
        "Coolant Temperature Fluctuation":      {"Radiator": 1.0, "Thermostat": 0.8, "Coolant Pump": 0.7},
        "Manifold Pressure Instability (MAP)":  {"MAP Sensor": 1.0, "Intake Manifold": 0.7, "Vacuum Lines": 0.6},
        "Fuel Supply / Injector Pressure Drop": {"Fuel Pump": 1.0, "Fuel Injector": 0.8, "Fuel Filter": 0.6}
    }
    sensor_to_cause = {
        "Battery Voltage (V)":      "Battery / Charging System Issue",
        "Coolant Temp (°C)":        "Coolant Temperature Fluctuation",
        "MAP (kPa)":                "Manifold Pressure Instability (MAP)",
        "Fuel Rail Pressure (bar)": "Fuel Supply / Injector Pressure Drop"
    }
    component_scores = {}
    for sensor, severity in sensor_severity.items():
        cause = sensor_to_cause.get(sensor)
        if cause in root_causes and cause in cause_component_map:
            for component, weight in cause_component_map[cause].items():
                component_scores[component] = component_scores.get(component, 0) + (severity * weight)

    if component_scores:
        total = sum(component_scores.values())
        return sorted([(c, round((v/total)*100, 1)) for c, v in component_scores.items()], key=lambda x: x[1], reverse=True)

    # --- Path 2: Direct sensor→component mapping for any CSV format ---
    # Based on SAE J1979 / automotive systems engineering relationships.
    # Each sensor's variance (severity score) is used to weight the components
    # it is physically responsible for monitoring.
    SENSOR_COMPONENT_DIRECT = {
        # Intake / Air system
        "Total Air Mass Flow Into Engine":       {"Air Filter": 1.0, "MAF Sensor": 0.9, "Turbocharger": 0.7, "Intake Manifold": 0.6},
        "Boost Pressure (mV)":                   {"Turbocharger": 1.0, "Intercooler": 0.8, "Wastegate": 0.7, "Boost Controller": 0.6},
        "Boost Temperature (kg/h)":              {"Turbocharger": 1.0, "Intercooler": 0.9, "Charge Air Cooler": 0.7},
        "HFM Temperature":                       {"MAF Sensor": 1.0, "Air Filter": 0.8, "Intake System": 0.6},
        "Ambient Pressure (bar)":                {"MAP Sensor": 0.8, "ECU / Fueling": 0.6},
        # Thermal / Cooling system
        "Coolant Temperature Raw":               {"Radiator": 1.0, "Thermostat": 0.9, "Coolant Pump": 0.7, "Water Pump": 0.6},
        "Coolant Temp (°C)":                     {"Radiator": 1.0, "Thermostat": 0.9, "Coolant Pump": 0.7},
        "Fuel Temperature":                      {"Fuel Cooler": 1.0, "Fuel Injector": 0.8, "Fuel Filter": 0.7, "Fuel System": 0.6},
        "Ambient Temperature (degree C)":        {"Engine Cooling": 0.7, "HVAC System": 0.6, "Radiator": 0.5},
        "Cabin Temperature":                     {"HVAC System": 1.0, "Heater Core": 0.8, "AC Compressor": 0.7},
        # Electrical / Charging
        "Battery Voltage (V)":                   {"Battery": 1.0, "Alternator": 0.8, "Voltage Regulator": 0.6},
        # Fuel system
        "Fuel Rail Pressure (bar)":              {"Fuel Pump": 1.0, "Fuel Injector": 0.8, "Fuel Filter": 0.6},
        # Drivetrain / Engine
        "Engine RPM":                            {"Crankshaft Bearings": 1.0, "Timing Chain": 0.8, "Engine Mount": 0.6},
        "Vehicle Speed (km/h)":                  {"Drivetrain": 1.0, "Transmission": 0.8, "Wheel Bearings": 0.6},
        "Throttle Position (%)":                 {"Throttle Body": 1.0, "Accelerator Pedal": 0.8, "Drive-by-Wire": 0.6},
        "MAP (kPa)":                             {"MAP Sensor": 1.0, "Intake Manifold": 0.8, "Vacuum Lines": 0.6},
        "Accelerator Pedal Position (%)":        {"Throttle Body": 1.0, "Accelerator Pedal": 0.9, "Drive-by-Wire": 0.7},
    }
    for sensor, severity in sensor_severity.items():
        if severity <= 0:
            continue
        components = SENSOR_COMPONENT_DIRECT.get(sensor, {})
        for component, weight in components.items():
            component_scores[component] = component_scores.get(component, 0) + (severity * weight)

    if not component_scores:
        return []
    total = sum(component_scores.values())
    return sorted([(c, round((v/total)*100, 1)) for c, v in component_scores.items()], key=lambda x: x[1], reverse=True)[:6]

def generate_swot(sensor_severity, component_confidence, climate_warnings):
    s, w, o, t = [], [], ["Climate-adaptive maintenance plan"], []
    for sensor, val in sensor_severity.items():
        if val < 20: s.append(f"{sensor} stable ({val}%)")
        elif val > 40: w.append(f"{sensor} anomaly ({val}%)")
    t.extend(climate_warnings)
    if component_confidence: t.append(f"{component_confidence[0][0]} failure risk ({component_confidence[0][1]}%)")
    return {"strengths": s or ["Electrical stability"], "weaknesses": w or ["None detected"], "opportunities": o, "threats": t or ["No immediate threats"]}

# ================================================================
# NEW 1: Driver Behavior Scoring
# Uses: RPM, Throttle Position (%), Vehicle Speed, MAP
# Detects: harsh braking, rapid acceleration, over-revving
# ================================================================
def compute_driver_behavior_score(df, vehicle_type=None):
    """
    Scores driving behavior 0–100 (100 = perfect, 0 = very aggressive).
    Penalties applied for:
      - Over-revving: RPM > redline (varies by vehicle type)
      - Rapid acceleration: Throttle spike > 70% in <3 rows
      - Harsh braking: Speed drop > 20 km/h in <3 rows
      - High MAP sustained: MAP > threshold (varies by vehicle type) for >10% of trip
    """
    score = 100.0
    events = []

    # Get vehicle profile thresholds
    profile = get_vehicle_profile(vehicle_type)
    rpm_redline = profile["rpm_redline"]
    map_threshold = profile["map_overload_threshold"]

    # --- Over-revving ---
    if "Engine RPM" in df.columns:
        over_rev_pct = (df["Engine RPM"] > rpm_redline).mean() * 100
        penalty = min(30, over_rev_pct * 1.5)
        score -= penalty
        if over_rev_pct > 5:
            events.append(f"Over-revving detected ({round(over_rev_pct, 1)}% of trip above 4500 RPM)")

    # --- Rapid acceleration (Throttle spikes) ---
    if "Throttle Position (%)" in df.columns:
        throttle_diff = df["Throttle Position (%)"].diff().abs()
        rapid_accel_count = (throttle_diff > 30).sum()
        penalty = min(25, rapid_accel_count * 0.5)
        score -= penalty
        if rapid_accel_count > 5:
            events.append(f"Rapid acceleration events: {int(rapid_accel_count)} times (throttle spike >30%)")

    # --- Harsh braking (Speed drops) ---
    if "Vehicle Speed (km/h)" in df.columns:
        speed_diff = df["Vehicle Speed (km/h)"].diff()
        harsh_brake_count = (speed_diff < -20).sum()
        penalty = min(25, harsh_brake_count * 1.0)
        score -= penalty
        if harsh_brake_count > 3:
            events.append(f"Harsh braking events: {int(harsh_brake_count)} times (speed drop >20 km/h)")

    # --- Sustained high MAP (engine overload) ---
    if "MAP (kPa)" in df.columns:
        high_map_pct = (df["MAP (kPa)"] > map_threshold).mean() * 100
        penalty = min(20, high_map_pct * 0.8)
        score -= penalty
        if high_map_pct > 10:
            events.append(f"Sustained engine overload ({round(high_map_pct, 1)}% of trip MAP >{map_threshold} kPa)")

    score = max(0, round(score, 1))

    # Grade
    if score >= 85:   grade = "A — Excellent"
    elif score >= 70: grade = "B — Good"
    elif score >= 55: grade = "C — Average"
    elif score >= 40: grade = "D — Poor"
    else:             grade = "F — Dangerous"

    # Wear multiplier: aggressive driving accelerates component wear
    wear_multiplier = round(1.0 + ((100 - score) / 100) * 0.5, 3)

    return {
        "score": score,
        "grade": grade,
        "wear_multiplier": wear_multiplier,
        "events": events if events else ["No aggressive driving patterns detected"],
        "over_rev_pct": round((df["Engine RPM"] > 4500).mean() * 100, 1) if "Engine RPM" in df.columns else 0,
        "rapid_accel_count": int((df["Throttle Position (%)"].diff().abs() > 30).sum()) if "Throttle Position (%)" in df.columns else 0,
        "harsh_brake_count": int((df["Vehicle Speed (km/h)"].diff() < -20).sum()) if "Vehicle Speed (km/h)" in df.columns else 0,
    }

# ================================================================
# NEW 2: Remaining Useful Life (RUL) per Component
# Uses linear degradation trend on sensor rolling std/mean
# ================================================================
def compute_rul_per_component(df, wear_multiplier=1.0, last_serviced=None):
    """
    Estimates RUL in days for each component using degradation trend.
    Method: fit linear regression on sensor degradation proxy over time,
    extrapolate to failure threshold. Adjusted by driver wear multiplier.

    If last_serviced dict is provided, resets RUL baseline for recently serviced components.
    """
    results = {}
    n = len(df)
    time_index = np.arange(n)

    components = {
        "Battery": {
            "sensor": "Battery Voltage (V)",
            "proxy": lambda s: s.rolling(10).std().fillna(0),
            "threshold": 0.8,   # std > 0.8 = failure
            "base_days": 30,
        },
        "Radiator / Coolant": {
            "sensor": "Coolant Temp (°C)",
            "proxy": lambda s: s.rolling(10).std().fillna(0),
            "threshold": 10.0,
            "base_days": 45,
        },
        "MAP Sensor": {
            "sensor": "MAP (kPa)",
            "proxy": lambda s: s.rolling(10).std().fillna(0),
            "threshold": 15.0,
            "base_days": 60,
        },
        "Fuel Pump": {
            "sensor": "Fuel Rail Pressure (bar)",
            # After kPa→bar conversion, healthy range is 3–7 bar. Flag if mean drops below 2.5 bar.
            "proxy": lambda s: (2.5 - s.rolling(10).mean().fillna(5.0)).clip(lower=0),
            "threshold": 1.0,
            "base_days": 40,
        },
    }

    for component, config in components.items():
        sensor = config["sensor"]
        if sensor not in df.columns:
            continue

        series = config["proxy"](df[sensor]).values
        threshold = config["threshold"]
        current_val = series[-1] if len(series) > 0 else 0

        # Linear regression on last 50 points to get trend
        window = min(50, len(series))
        x = np.arange(window)
        y = series[-window:]

        try:
            slope, intercept, r_value, _, _ = linregress(x, y)
        except Exception:
            slope, intercept = 0, current_val

        # Rows until threshold at current trend
        if slope > 0 and current_val < threshold:
            rows_to_failure = (threshold - intercept) / slope
            rows_remaining = max(0, rows_to_failure - len(series))
            # Convert rows to days (assuming ~500 rows per day of driving data)
            days_remaining = rows_remaining / 500
        elif current_val >= threshold:
            days_remaining = 0
        else:
            days_remaining = config["base_days"]  # flat trend = use default

        # If component was recently serviced, reset RUL to base_days
        if last_serviced and component in last_serviced:
            from datetime import date
            try:
                svc_date = date.fromisoformat(last_serviced[component]["service_date"])
                days_since_service = (date.today() - svc_date).days
                # Proportionally restore RUL: if serviced recently, start from base_days
                restored = max(0, config["base_days"] - days_since_service * 0.3)
                days_adjusted = round(restored / wear_multiplier, 1)
                days_adjusted = min(days_adjusted, config["base_days"])
            except Exception:
                pass
        else:
            # Adjust by driver wear multiplier
            days_adjusted = round(days_remaining / wear_multiplier, 1)
            days_adjusted = min(days_adjusted, config["base_days"])

        if days_adjusted <= 3:      urgency = "CRITICAL"
        elif days_adjusted <= 10:   urgency = "WARNING"
        else:                       urgency = "HEALTHY"

        results[component] = {
            "rul_days": days_adjusted,
            "urgency": urgency,
            "current_degradation": round(float(current_val), 3),
            "trend_slope": round(float(slope), 5),
            "r_squared": round(float(r_value**2), 3) if 'r_value' in dir() else 0,
        }

    return results

# ================================================================
# NEW 3: Anomaly Scrubber Data
# Returns per-row sensor data with anomaly flags for the timeline slider
# ================================================================
def build_scrubber_data(df):
    """
    Returns per-row sensor data for the frontend timeline + sensor chart.
    Includes ALL numeric columns — works with any vehicle CSV regardless of column names.
    Standard sensors are listed first; unrecognised columns are appended after.
    """
    standard = ["Coolant Temp (°C)", "MAP (kPa)", "Battery Voltage (V)",
                "Fuel Rail Pressure (bar)", "Engine RPM", "Vehicle Speed (km/h)",
                "Throttle Position (%)"]
    exclude  = {"Timestamp (s)", "timestamp", "index", "row_id", "Unnamed: 0"}

    # All numeric columns in the dataframe
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    all_numeric = [c for c in all_numeric if c not in exclude]

    # Standard sensors first (if present), then any extra columns
    available = [s for s in standard if s in all_numeric]
    for c in all_numeric:
        if c not in available:
            available.append(c)

    # Z-score anomaly flag per row (across all available sensors)
    if available:
        numeric = df[available].copy()
        z = (numeric - numeric.mean()) / (numeric.std() + 1e-6)
        anomaly_flag = (z.abs() > 2.5).any(axis=1).astype(int).tolist()
    else:
        anomaly_flag = [0] * len(df)

    # Downsample to max 500 points for frontend performance
    step    = max(1, len(df) // 500)
    indices = list(range(0, len(df), step))

    if "Timestamp (s)" in df.columns:
        timestamps = [round(float(df["Timestamp (s)"].iloc[i]), 1) for i in indices]
    else:
        timestamps = list(range(len(indices)))

    result = {
        "timestamps":    timestamps,
        "anomaly_flags": [anomaly_flag[i] for i in indices],
        "sensors":       {}
    }
    for sensor in available:
        result["sensors"][sensor] = [round(float(df[sensor].iloc[i]), 2) for i in indices]

    return result

# ================================================================
# NEW 4: Alert System (Twilio WhatsApp + SendGrid Email)
# Plug in your keys in .env — structure is ready
# ================================================================
def send_whatsapp_alert(risk, status, top_component, file_name):
    """
    Sends WhatsApp alert via Twilio when status = CRITICAL.
    Add to .env:
        TWILIO_ACCOUNT_SID=ACxxxx
        TWILIO_AUTH_TOKEN=xxxx
        TWILIO_FROM=whatsapp:+14155238886
        ALERT_TO_WHATSAPP=whatsapp:+91xxxxxxxxxx
    """
    sid   = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_ = os.getenv("TWILIO_FROM")
    to_   = os.getenv("ALERT_TO_WHATSAPP")

    if not all([sid, token, from_, to_]):
        print("⚠️  Twilio keys not configured. Skipping WhatsApp alert.")
        return False

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        message = client.messages.create(
            body=(
                f"🚨 VEHICLE ALERT — {status}\n"
                f"File: {file_name}\n"
                f"Risk Score: {risk}\n"
                f"Top Component at Risk: {top_component}\n"
                f"Immediate inspection recommended."
            ),
            from_=from_,
            to=to_
        )
        print(f"WhatsApp alert sent: {message.sid}")
        return True
    except Exception as e:
        print(f"WhatsApp alert failed: {e}")
        return False

def send_email_alert(risk, status, top_component, file_name, root_causes):
    """
    Sends email alert via Gmail SMTP when status = CRITICAL.
    Add to .env:
        ALERT_EMAIL_FROM=your@gmail.com
        ALERT_EMAIL_PASSWORD=your_app_password   (Gmail App Password)
        ALERT_EMAIL_TO=manager@company.com
    """
    from_email = os.getenv("ALERT_EMAIL_FROM")
    password   = os.getenv("ALERT_EMAIL_PASSWORD")
    to_email   = os.getenv("ALERT_EMAIL_TO")

    if not all([from_email, password, to_email]):
        print("⚠️  Email keys not configured. Skipping email alert.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 Vehicle CRITICAL Alert — Risk {risk} | {file_name}"
        msg["From"]    = from_email
        msg["To"]      = to_email

        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;background:#141e22;color:#fff;padding:20px;">
        <h2 style="color:#e74c3c;">🚨 Vehicle Predictive Maintenance Alert</h2>
        <table style="border-collapse:collapse;width:100%;">
          <tr><td style="padding:8px;color:#aaa;">Status</td>
              <td style="padding:8px;color:#e74c3c;font-weight:bold;">{status}</td></tr>
          <tr><td style="padding:8px;color:#aaa;">Risk Score</td>
              <td style="padding:8px;font-weight:bold;">{risk}</td></tr>
          <tr><td style="padding:8px;color:#aaa;">File</td>
              <td style="padding:8px;">{file_name}</td></tr>
          <tr><td style="padding:8px;color:#aaa;">Top Component</td>
              <td style="padding:8px;color:#f39c12;font-weight:bold;">{top_component}</td></tr>
          <tr><td style="padding:8px;color:#aaa;">Root Causes</td>
              <td style="padding:8px;">{', '.join(root_causes)}</td></tr>
        </table>
        <p style="color:#aaa;margin-top:20px;">Immediate inspection is recommended.</p>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print(f"Email alert sent to {to_email}")
        return True
    except Exception as e:
        print(f"Email alert failed: {e}")
        return False

# ================================================================
# AUTH ENDPOINTS
# ================================================================
@app.post("/auth/register")
def register(user: UserRegister):
    try:
        hashed = get_password_hash(user.password)
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s) RETURNING id",
            (user.username, user.email, hashed)
        )
        user_id = cursor.fetchone()[0]
        conn.commit()
        token = create_access_token({"user_id": user_id, "username": user.username})
        return {"token": token, "user_id": user_id, "username": user.username}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
def login(user: UserLogin):
    try:
        cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = %s", (user.username,))
        row = cursor.fetchone()
        if not row or not verify_password(user.password, row[2]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token = create_access_token({"user_id": row[0], "username": row[1]})
        return {"token": token, "user_id": row[0], "username": row[1]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/me")
def get_me(user_id: int = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = %s", (user_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": row[0], "username": row[1], "email": row[2], "created_at": str(row[3])}

# ================================================================
# VEHICLE MANAGEMENT ENDPOINTS
# ================================================================
@app.post("/vehicles")
def create_vehicle(vehicle: VehicleCreate, user_id: int = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    try:
        cursor.execute(
            "INSERT INTO vehicles (user_id, name, vehicle_type, year, make, model_name) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (user_id, vehicle.name, vehicle.vehicle_type, vehicle.year, vehicle.make, vehicle.model_name)
        )
        vehicle_id = cursor.fetchone()[0]
        conn.commit()
        return {"vehicle_id": vehicle_id, "name": vehicle.name, "vehicle_type": vehicle.vehicle_type}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/vehicles")
def list_vehicles(user_id: int = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    cursor.execute(
        "SELECT id, name, vehicle_type, year, make, model_name, created_at FROM vehicles WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    return [{"id": r[0], "name": r[1], "vehicle_type": r[2], "year": r[3], "make": r[4], "model_name": r[5], "created_at": str(r[6])} for r in rows]

@app.delete("/vehicles/{vehicle_id}")
def delete_vehicle(vehicle_id: int, user_id: int = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    cursor.execute("DELETE FROM vehicles WHERE id = %s AND user_id = %s", (vehicle_id, user_id))
    conn.commit()
    return {"deleted": vehicle_id}

# ================================================================
# MAINTENANCE LOG ENDPOINTS
# ================================================================
@app.post("/maintenance/log")
def log_maintenance(
    vehicle_id: int = Form(...),
    component: str = Form(...),
    service_date: str = Form(...),
    notes: str = Form(""),
    user_id: int = Depends(get_current_user)
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    try:
        cursor.execute(
            "INSERT INTO maintenance_log (user_id, vehicle_id, component, service_date, notes) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (user_id, vehicle_id, component, service_date, notes)
        )
        log_id = cursor.fetchone()[0]
        conn.commit()
        return {"log_id": log_id, "message": f"{component} serviced on {service_date}. RUL reset."}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/maintenance/history/{vehicle_id}")
def get_maintenance_history(vehicle_id: int, user_id: int = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    cursor.execute(
        "SELECT id, component, service_date, notes, logged_at FROM maintenance_log WHERE vehicle_id = %s AND user_id = %s ORDER BY service_date DESC",
        (vehicle_id, user_id)
    )
    rows = cursor.fetchall()
    return [{"id": r[0], "component": r[1], "service_date": str(r[2]), "notes": r[3], "logged_at": str(r[4])} for r in rows]

@app.get("/maintenance/last-service/{vehicle_id}")
def get_last_service(vehicle_id: int, user_id: int = Depends(get_current_user)):
    """Returns most recent service date per component — used to reset RUL baseline."""
    if not user_id:
        return {}
    cursor.execute(
        """SELECT DISTINCT ON (component) component, service_date, notes
           FROM maintenance_log WHERE vehicle_id = %s AND user_id = %s
           ORDER BY component, service_date DESC""",
        (vehicle_id, user_id)
    )
    rows = cursor.fetchall()
    return {r[0]: {"service_date": str(r[1]), "notes": r[2]} for r in rows}

# ================================================================
# SCHEDULED REPORTS — Weekly email every Monday 8am
# ================================================================
scheduler = AsyncIOScheduler()

async def send_weekly_reports():
    """Sends weekly health summary email to all registered users who have vehicles."""
    from_email = os.getenv("ALERT_EMAIL_FROM")
    password   = os.getenv("ALERT_EMAIL_PASSWORD")
    if not all([from_email, password]):
        print("⚠️  Email not configured, skipping weekly report.")
        return
    try:
        cursor.execute("""
            SELECT u.email, u.username, v.name, v.id,
                   p.risk, p.status, p.recommended_components, p.upload_time
            FROM users u
            JOIN vehicles v ON v.user_id = u.id
            LEFT JOIN LATERAL (
                SELECT risk, status, recommended_components, upload_time
                FROM predictions WHERE vehicle_id = v.id
                ORDER BY upload_time DESC LIMIT 1
            ) p ON true
            WHERE u.email IS NOT NULL
        """)
        rows = cursor.fetchall()
        if not rows:
            return
        # Group by user email
        from collections import defaultdict
        user_reports = defaultdict(list)
        for row in rows:
            email, username, vname, vid, risk, vstatus, comps, uptime = row
            user_reports[(email, username)].append({
                "vehicle": vname, "risk": risk, "status": vstatus,
                "components": comps, "last_upload": str(uptime) if uptime else "Never"
            })
        for (email, username), vehicles in user_reports.items():
            rows_html = ""
            for v in vehicles:
                color = "#e74c3c" if v["status"] == "CRITICAL" else "#f39c12" if v["status"] == "WARNING" else "#2ecc71"
                rows_html += f"""
                <tr>
                  <td style="padding:8px;border-bottom:1px solid #2a3a42;">{v['vehicle']}</td>
                  <td style="padding:8px;border-bottom:1px solid #2a3a42;color:{color};font-weight:bold;">{v['status'] or 'N/A'}</td>
                  <td style="padding:8px;border-bottom:1px solid #2a3a42;">{round(v['risk'],3) if v['risk'] else 'N/A'}</td>
                  <td style="padding:8px;border-bottom:1px solid #2a3a42;">{v['last_upload']}</td>
                </tr>"""
            html = f"""
            <html><body style="font-family:Arial;background:#141e22;color:#fff;padding:20px;">
            <h2 style="color:#00d2ff;">🚗 Weekly Vehicle Health Report</h2>
            <p>Hi {username}, here's your fleet summary for this week:</p>
            <table style="width:100%;border-collapse:collapse;">
              <tr style="background:#1e292e;">
                <th style="padding:10px;text-align:left;">Vehicle</th>
                <th style="padding:10px;text-align:left;">Status</th>
                <th style="padding:10px;text-align:left;">Risk Score</th>
                <th style="padding:10px;text-align:left;">Last Upload</th>
              </tr>
              {rows_html}
            </table>
            <p style="color:#aab8c2;margin-top:20px;">Open your dashboard to view full details.</p>
            </body></html>"""
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = "🚗 Weekly Vehicle Health Report"
                msg["From"] = from_email
                msg["To"] = email
                msg.attach(MIMEText(html, "html"))
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(from_email, password)
                    server.sendmail(from_email, email, msg.as_string())
                print(f"Weekly report sent to {email}")
            except Exception as e:
                print(f"Failed to send report to {email}: {e}")
    except Exception as e:
        print(f"Weekly report job failed: {e}")

@app.on_event("startup")
async def startup_event():
    scheduler.add_job(send_weekly_reports, "cron", day_of_week="mon", hour=8, minute=0)
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

# ================================================================
# NEW 5: WebSocket — Simulated OBD-II Live Stream
# Streams fake real-time sensor data every second
# In production: replace random values with actual OBD-II reads
# ================================================================
@app.websocket("/ws/obd-live")
async def obd_live_stream(websocket: WebSocket):
    await websocket.accept()
    print("OBD-II WebSocket connected")
    try:
        # Simulation state
        rpm        = 800.0
        speed      = 0.0
        coolant    = 75.0
        map_kpa    = 35.0
        battery    = 14.2
        fuel_pr    = 55.0
        throttle   = 5.0
        tick       = 0

        while True:
            tick += 1

            # Simulate realistic sensor drift + occasional spikes
            rpm      = max(700, min(6000, rpm + random.gauss(0, 80)))
            speed    = max(0,   min(180,  speed + random.gauss(0, 3)))
            coolant  = max(60,  min(115,  coolant + random.gauss(0, 0.3)))
            map_kpa  = max(20,  min(105,  map_kpa + random.gauss(0, 1.5)))
            battery  = max(11,  min(15,   battery + random.gauss(0, 0.05)))
            fuel_pr  = max(20,  min(80,   fuel_pr + random.gauss(0, 0.8)))
            throttle = max(0,   min(100,  throttle + random.gauss(0, 4)))

            # Inject anomaly spikes every 30 seconds for demo effect
            if tick % 30 == 0:
                rpm     += random.choice([-800, 1200])
                coolant += random.choice([8, -5])
                battery -= random.uniform(0.5, 1.5)

            # Quick live risk estimate (simplified heuristic for speed)
            live_risk = round(
                min(1.0, 0.2 +
                    (max(0, coolant - 95) * 0.008) +
                    (max(0, rpm - 4500) * 0.00005) +
                    (max(0, 12.0 - battery) * 0.05) +
                    (max(0, 35 - fuel_pr) * 0.005)
                ), 3
            )

            payload = {
                "tick": tick,
                "timestamp_s": tick,
                "Engine RPM":               round(rpm, 0),
                "Vehicle Speed (km/h)":     round(speed, 1),
                "Coolant Temp (°C)":        round(coolant, 1),
                "MAP (kPa)":                round(map_kpa, 1),
                "Battery Voltage (V)":      round(battery, 2),
                "Fuel Rail Pressure (bar)": round(fuel_pr, 1),
                "Throttle Position (%)":    round(throttle, 1),
                "live_risk":                live_risk,
                "status": "CRITICAL" if live_risk > 0.8 else "WARNING" if live_risk > 0.5 else "SAFE"
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)  # 1 reading per second

    except WebSocketDisconnect:
        print("OBD-II WebSocket disconnected")
    except Exception as e:
        print(f"OBD-II WebSocket error: {e}")

# ================================================================
# NEW 6: Multi-File Comparison Endpoint
# Accepts multiple CSVs, returns per-file risk + sensor trends
# ================================================================
@app.post("/compare")
async def compare_files(files: List[UploadFile] = File(...)):
    """
    Accepts 2–10 CSV files (e.g. Day_01.csv to Day_07.csv).
    Returns per-file: risk, status, driver score, RUL snapshot, sensor averages.
    Used for the multi-vehicle/multi-day comparison chart.
    """
    results = []
    try:
        for f in files:
            contents = await f.read()
            df = pd.read_csv(BytesIO(contents))
            df.columns = df.columns.str.strip()
            df = normalize_columns(df)   # apply same column aliasing as /predict
            # CSV stores Fuel Rail Pressure in kPa despite column name saying "bar" — convert to real bar
            if "Fuel Rail Pressure (bar)" in df.columns:
                df["Fuel Rail Pressure (bar)"] = df["Fuel Rail Pressure (bar)"] / 100.0

            X_raw = build_features(df).dropna()
            if len(X_raw) >= 5:
                risk_series = model.predict(X_raw[model_features])
                base_risk   = float(risk_series[-1])
            else:
                # Same physics-based fallback as /predict (ISO 13381-1 z-score method)
                exclude_ts = {"Timestamp (s)", "timestamp", "index", "row_id", "Unnamed: 0"}
                num_df = df.select_dtypes(include=[np.number])
                num_df = num_df[[c for c in num_df.columns if c not in exclude_ts]]
                if num_df.shape[1] > 0:
                    z_all      = (num_df - num_df.mean()) / (num_df.std() + 1e-6)
                    row_risk   = (z_all.abs().mean(axis=1) / 3.0).clip(0, 1)
                    risk_series = row_risk.rolling(window=20, min_periods=1).mean().values
                else:
                    risk_series = np.array([0.3])
                base_risk = float(risk_series[-1])
            stress_mult, final_t, final_h, _ = calculate_environmental_stress(df)
            final_risk  = round(min(1.0, base_risk * stress_mult), 3)
            status      = classify_risk_dynamic(final_risk, risk_series)
            severity    = compute_sensor_severity(df)
            causes      = detect_root_cause(df)
            driver      = compute_driver_behavior_score(df)
            rul         = compute_rul_per_component(df, driver["wear_multiplier"])

            results.append({
                "file":           f.filename,
                "risk":           final_risk,
                "status":         status,
                "ambient_temp":   round(final_t, 1),
                "humidity":       round(final_h, 1),
                "sensor_severity": severity,
                "root_causes":    causes,
                "driver_score":   driver["score"],
                "driver_grade":   driver["grade"],
                "rul_snapshot": {
                    comp: data["rul_days"] for comp, data in rul.items()
                },
                # Average sensor values for comparison radar/bar chart
                "sensor_averages": {
                    col: round(float(df[col].mean()), 2)
                    for col in ["Coolant Temp (°C)", "MAP (kPa)",
                                "Battery Voltage (V)", "Fuel Rail Pressure (bar)",
                                "Engine RPM", "Vehicle Speed (km/h)"]
                    if col in df.columns
                },
                "risk_trend": [round(float(x), 3) for x in risk_series[-50:]]  # last 50 pts
            })
    except Exception as e:
        return {"error": str(e)}

    return sanitize_floats({"comparison": results})

# ================================================================
# EXISTING + UPDATED: /predict endpoint
# Added: driver score, RUL, scrubber data, alert trigger
# ================================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    city: str = Form(None),
    timestamp: str = Form(None),
    vehicle_id: int = Form(None),
    vehicle_type: str = Form(None),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        df.columns = df.columns.str.strip()
        # Auto-map real-world column name variants to internal standard names
        df = normalize_columns(df)
        user_id = get_current_user(credentials) if credentials else None
        # CSV stores Fuel Rail Pressure in kPa despite column name saying "bar" — convert to real bar
        if "Fuel Rail Pressure (bar)" in df.columns:
            df["Fuel Rail Pressure (bar)"] = df["Fuel Rail Pressure (bar)"] / 100.0

        X_raw = build_features(df).dropna()
        if len(X_raw) >= 5:
            # Standard path: ML model on known features
            risk_series = model.predict(X_raw[model_features])
            base_risk   = float(risk_series[-1])
        else:
            # Physics-based fallback for CSVs without the 4 ML columns.
            # Per-row rolling z-score anomaly magnitude across all numeric sensors
            # (grounded in ISO 13381-1 condition monitoring methodology).
            exclude_ts  = {"Timestamp (s)", "timestamp", "index", "row_id", "Unnamed: 0"}
            num_df = df.select_dtypes(include=[np.number])
            num_df = num_df[[c for c in num_df.columns if c not in exclude_ts]]
            if num_df.shape[1] > 0:
                z_all      = (num_df - num_df.mean()) / (num_df.std() + 1e-6)
                row_risk   = (z_all.abs().mean(axis=1) / 3.0).clip(0, 1)
                # Rolling window smoothing — reduces noise while preserving trend
                risk_series = row_risk.rolling(window=20, min_periods=1).mean().values
            else:
                risk_series = np.array([0.3])
            base_risk = float(risk_series[-1])

        actual_temp, actual_hum = get_visual_crossing_weather(city, timestamp)
        stress_mult, final_t, final_h, warnings = calculate_environmental_stress(df, actual_temp, actual_hum)
        final_risk  = min(1.0, base_risk * stress_mult)
        status      = classify_risk_dynamic(final_risk, risk_series)
        ttf         = (1 - final_risk) * HORIZON

        severity   = compute_sensor_severity(df)
        causes     = detect_root_cause(df)
        comp_conf  = map_components_with_confidence(causes, severity)
        swot       = generate_swot(severity, comp_conf, warnings)

        climate_impact_pct  = quantify_climate_impact(base_risk, final_risk)
        extreme_analysis    = get_extreme_condition_analysis(final_t, final_h, base_risk)
        contribution_split  = compute_climate_vs_internal_split(base_risk, final_risk, severity)

        # --- NEW: Driver Behavior ---
        driver_behavior = compute_driver_behavior_score(df, vehicle_type=vehicle_type)

        # --- Get last service dates for RUL reset ---
        last_service_dict = {}
        if user_id and vehicle_id:
            cursor.execute(
                """SELECT DISTINCT ON (component) component, service_date, notes
                   FROM maintenance_log WHERE vehicle_id = %s AND user_id = %s
                   ORDER BY component, service_date DESC""",
                (vehicle_id, user_id)
            )
            service_rows = cursor.fetchall()
            last_service_dict = {r[0]: {"service_date": str(r[1]), "notes": r[2]} for r in service_rows}

        # --- NEW: RUL per component (adjusted by driver wear and maintenance history) ---
        rul = compute_rul_per_component(df, driver_behavior["wear_multiplier"], last_serviced=last_service_dict)

        # --- NEW: Scrubber data ---
        scrubber = build_scrubber_data(df)

        # Database log — cast to Python float to avoid psycopg2 "schema np does not exist" error
        cursor.execute("""
            INSERT INTO predictions (file_name, risk, status, time_to_fault, root_causes, recommended_components, user_id, vehicle_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (file.filename, float(final_risk), status, float(ttf),
              ", ".join(causes),
              ", ".join([f"{c}({p}%)" for c, p in comp_conf]),
              user_id, vehicle_id))
        conn.commit()

        # --- NEW: Fire alerts if CRITICAL ---
        alert_sent = {"whatsapp": False, "email": False}
        if status == "CRITICAL":
            top_comp = comp_conf[0][0] if comp_conf else "Unknown"
            alert_sent["whatsapp"] = send_whatsapp_alert(round(final_risk, 3), status, top_comp, file.filename)
            alert_sent["email"]    = send_email_alert(round(final_risk, 3), status, top_comp, file.filename, causes)

        return sanitize_floats({
            # Existing fields
            "status":           status,
            "risk":             round(final_risk, 3),
            "base_risk":        round(base_risk, 3),
            "climate_stress_pct": round((stress_mult - 1) * 100, 1),
            "ambient_temp":     round(final_t, 1),
            "humidity":         round(final_h, 1),
            "time_to_fault":    round(ttf, 2),
            "anomalies":        detect_anomalies(df),
            "risk_trend":       [round(float(x), 3) for x in risk_series],
            "time_axis":        list(range(len(risk_series))),
            "root_causes":      causes,
            "sensor_severity":  severity,
            "recommended_components": comp_conf,
            "swot":             swot,
            "climate_impact_pct":  climate_impact_pct,
            "extreme_analysis":    extreme_analysis,
            "contribution_split":  contribution_split,

            # New fields
            "driver_behavior":  driver_behavior,
            "rul":              rul,
            "scrubber":         scrubber,
            "alert_sent":       alert_sent,
            "vehicle_profile":  get_vehicle_profile(vehicle_type),
        })
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history():
    cursor.execute("""
        SELECT file_name, upload_time, risk, status, root_causes, recommended_components
        FROM predictions ORDER BY upload_time DESC LIMIT 7
    """)
    return [
        {
            "file_name": r[0],
            "upload_time": r[1].strftime("%Y-%m-%d %H:%M"),
            "risk": float(r[2]),
            "status": r[3],
            "root_causes": r[4],
            "recommended_components": r[5]
        }
        for r in cursor.fetchall()
    ]