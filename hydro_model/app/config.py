from pydantic import BaseModel
from pathlib import Path

class Config(BaseModel):
    model_dir: Path
    model_name: str
    x_scaler_name: str
    y_scaler_name: str
    sample_time: int  # in minutes
    X_lag: int  # this times 10 gives the number of minutes of historical data used for each prediction
    y_lag: int  # this times 10 gives the number of minutes of historical data used for each prediction