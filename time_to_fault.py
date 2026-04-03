# Convert risk to time to fault
from app import HORIZON, risk


time_to_fault = float((1 - risk) * HORIZON)