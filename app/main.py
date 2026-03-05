import pickle
from pathlib import Path

from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS

app = FastAPI()

# Load model
with open("/app/saved_models/narx_model.pkl", "rb") as f:
    model: FROLS = pickle.load(f)

# Load scalers
x_scaler: StandardScaler = joblib.load("/app/saved_models/x_scaler.pkl")
y_scaler: StandardScaler = joblib.load("/app/saved_models/y_scaler.pkl")

# Load combined data for historical context
combined_data_path = "/app/data/paired/combined_data.csv"
combined_df = pd.read_csv(combined_data_path, parse_dates=['Date'])


@app.post("/predict")
def predict(rain_1: float, rain_2: float):
    """
    Make a prediction using the NARX model.
    
    The NARX model requires lag=100, so we need historical data to provide context.
    This function:
    1. Gets the last 100 timesteps from combined data
    2. Appends the new rain values as the next timestep
    3. Provides this context to the model for prediction
    4. Returns the predicted streamflow for the new timestep
    """
    
    # Extract rain columns and scale historical data
    rain_columns = [col for col in combined_df.columns if "rain" in col.lower()]
    target_column = "Stage"
    
    # Get last 100 timesteps from combined data for NARX context
    X_history = combined_df[rain_columns].tail(100).values
    y_history = combined_df[target_column].tail(100).values.reshape(-1, 1)
    
    # Scale historical data
    X_history_scaled = x_scaler.transform(X_history)
    y_history_scaled = y_scaler.transform(y_history)
    
    # Create new input with rain values and scale them
    new_input = np.array([[rain_1, rain_2]])
    new_input_scaled = x_scaler.transform(new_input)
    
    # Append new input to history (now we have 101 timesteps)
    X_full = np.vstack([X_history_scaled, new_input_scaled])
    y_full = np.vstack([y_history_scaled, [[0]]])  # Placeholder for y at new timestep
    
    # Make prediction - the model will predict the stage at the new timestep
    y_pred_scaled = model.predict(X=X_full, y=y_full)
    
    # Only take the prediction for the last timestep (our new prediction)
    y_pred_new_scaled = y_pred_scaled[-1:]
    
    # Rescale prediction back to original units
    y_pred_new = y_scaler.inverse_transform(y_pred_new_scaled)
    
    return {"predicted_streamflow": float(y_pred_new[0][0])}