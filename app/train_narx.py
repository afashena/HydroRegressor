from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def merge_rain_gages():
    """
    Merge all synced rain-gages CSVs from data/paired into a single DataFrame.
    Keeps only one timestamp column and combines all rain amount columns.
    
    Returns:
        pd.DataFrame: Merged dataframe with Date and all rain columns
    """
    paired_dir = Path(__file__).parent.parent / 'data' / 'paired'
    
    # Find all synced rain-gages files
    rain_files = sorted(paired_dir.glob('rain-gages_*_synced.csv'))
    if not rain_files:
        raise RuntimeError(f"No synced rain-gages files found in {paired_dir}")
    
    print(f"Merging {len(rain_files)} rain-gages files...")
    
    # Load first file
    merged = pd.read_csv(rain_files[0], parse_dates=['Date'])
    site_id = rain_files[0].stem.replace('rain-gages_', '').replace('_synced', '')
    merged = merged.rename(columns={'Rain Amount (in)': f'rain_{site_id}'})
    
    # Merge remaining files
    for rf in rain_files[1:]:
        df = pd.read_csv(rf, parse_dates=['Date'])
        site_id = rf.stem.replace('rain-gages_', '').replace('_synced', '')
        df = df.rename(columns={'Rain Amount (in)': f'rain_{site_id}'})
        merged = merged.merge(df, on='Date', how='inner')
    
    print(f"Merged shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    
    return merged


def merge_rain_and_stream():
    """
    Merge all synced rain-gages CSVs with the synced stream-gages CSV.
    Returns a combined dataframe with Date, all rain columns, and Stage.
    
    Returns:
        pd.DataFrame: Combined dataframe with Date, rain columns, and Stage
    """
    paired_dir = Path(__file__).parent.parent / 'data' / 'paired'
    
    # Load stream data
    stream_files = list(paired_dir.glob('stream-gages_*_synced.csv'))
    if not stream_files:
        raise RuntimeError(f"No synced stream-gages file found in {paired_dir}")
    
    combined = pd.read_csv(stream_files[0], parse_dates=['Date'])
    
    # Load and merge rain files
    rain_files = sorted(paired_dir.glob('rain-gages_*_synced.csv'))
    for rf in rain_files:
        df = pd.read_csv(rf, parse_dates=['Date'])
        site_id = rf.stem.replace('rain-gages_', '').replace('_synced', '')
        df = df.rename(columns={'Rain Amount (in)': f'rain_{site_id}'})
        combined = combined.merge(df, on='Date', how='inner')
    
    print(f"Combined data shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")

    combined.to_csv(paired_dir / "combined_data.csv", index=False)
    
    return combined


def preprocess_data(csv_path: Path, train_split: float):

    # -------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------

    # merge all synced CSVs into a single DataFrame
    df = merge_rain_and_stream()

    # Drop timestamp if present
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # -------------------------------------------------
    # 2. Separate Inputs (Rain Sensors) and Output
    # -------------------------------------------------

    rain_columns = [col for col in df.columns if "rain" in col.lower()]
    target_column = "Stage"

    X = df[rain_columns].values
    y = df[target_column].values.reshape(-1, 1)

    # -------------------------------------------------
    # 3. Scale Data (IMPORTANT for NARX stability)
    # -------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # -------------------------------------------------
    # 4. Train/Test Split (Time-Aware Split Recommended)
    # -------------------------------------------------

    split_index = int(len(X_scaled) * train_split)

    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]

    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index:]

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def train_model(X_train, y_train, x_scaler, y_scaler):

    # -------------------------------------------------
    # 5. Define NARX Model
    # -------------------------------------------------

    model = FROLS(
        order_selection=True,
        n_info_values=15,       # model complexity cap
        ylag=100,                 # past 100 storm drain values
        xlag=[100, 100],                 # past 100 rain values
        basis_function=Polynomial(degree=2),
    )

    # -------------------------------------------------
    # 6. Train Model
    # -------------------------------------------------

    model.fit(X=X_train, y=y_train)

    model_dir = Path(__file__).parent.parent / "saved_models"
    model_dir.mkdir(exist_ok=True)

    # Save the trained model
    with open(model_dir / "narx_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scalers
    with open(model_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)

    with open(model_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

def evaluate_model(X_train, y_train, X_test, y_test):
    # this function runs inference on the scaled test set

    model_dir = Path(__file__).parent.parent / "saved_models"

    # Load model
    with open(model_dir / "narx_model.pkl", "rb") as f:
        model: FROLS = pickle.load(f)

    # Load scalers
    with open(model_dir / "y_scaler.pkl", "rb") as f:
        y_scaler: StandardScaler = pickle.load(f)

    # For NARX with lag, we need to prepend the last lag values from train to test
    lag = 100  # ylag from model
    X_test_full = np.vstack([X_train[-lag:], X_test])
    y_test_full = np.vstack([y_train[-lag:], y_test])

    y_pred_full = model.predict(X=X_test_full, y=y_test_full)

    # Take only the predictions for the test set
    y_pred = y_pred_full[lag:]

    # Reverse scaling
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_test_rescaled = y_scaler.inverse_transform(y_test)

    # -------------------------------------------------
    # 8. Evaluate
    # -------------------------------------------------

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"MSE: {mse}")

    # -------------------------------------------------
    # 9. Plot Results
    # -------------------------------------------------

    plt.figure()
    plt.plot(y_test_rescaled, label="Actual")
    plt.plot(y_pred_rescaled, label="Predicted")
    plt.legend()
    plt.title("Storm Drain Prediction (NARX)")
    plt.grid(True)
    plt.savefig(model_dir / "narx_predictions.png", dpi=150)

def main(csv_path: Path, train_split: float):
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = preprocess_data(csv_path, train_split)
    train_model(X_train, y_train, x_scaler, y_scaler)
    evaluate_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    csv_path = Path("path_to_your_data.csv")
    train_split = 0.9
    main(csv_path, train_split)