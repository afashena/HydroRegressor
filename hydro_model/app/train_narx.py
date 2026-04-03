from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sysidentpy.parameter_estimation import LeastSquares, RidgeRegression

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.db_utils.pair_measurements import process_gage_data
from app.config import Config
from app.model import NeuralNARX

# Configurable rolling window sizes (in timesteps) for rolling sum features
ROLLING_WINDOWS = [3, 6, 12]  # Create rolling sums at these horizons

X_LAG = 10
Y_LAG = 10


def merge_rain_gages(rain_dfs: list[pd.DataFrame], site_ids: list[str]) -> pd.DataFrame:
    """
    Merge multiple rain-gage DataFrames into a single DataFrame.

    Args:
        rain_dfs (list[pd.DataFrame]): List of rain-gage DataFrames.
        site_ids (list[str]): List of site IDs corresponding to the DataFrames.

    Returns:
        pd.DataFrame: Merged DataFrame with Date and rain columns.
    """
    if not rain_dfs or not site_ids or len(rain_dfs) != len(site_ids):
        raise ValueError("Mismatch between rain DataFrames and site IDs.")

    print(f"Merging {len(rain_dfs)} rain-gages DataFrames...")

    # Start with the first DataFrame
    merged = rain_dfs[0].rename(columns={"rain_amount": f"rain_{site_ids[0]}"})

    # Merge remaining DataFrames
    for df, site_id in zip(rain_dfs[1:], site_ids[1:]):
        df = df.rename(columns={"rain_amount": f"rain_{site_id}"})
        merged = merged.merge(df, on="collect_date", how="inner")

    print(f"Merged shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")

    return merged


def merge_rain_and_stream(rain_dfs: list[pd.DataFrame], site_ids: list[str], stream_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rain-gage DataFrames with a stream-gage DataFrame.

    Args:
        rain_dfs (list[pd.DataFrame]): List of rain-gage DataFrames.
        site_ids (list[str]): List of site IDs corresponding to the rain DataFrames.
        stream_df (pd.DataFrame): Stream-gage DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame with Date, rain columns, and Stage.
    """
    if not rain_dfs or not site_ids or len(rain_dfs) != len(site_ids):
        raise ValueError("Mismatch between rain DataFrames and site IDs.")

    print("Merging rain-gages with stream-gages...")

    # Merge rain DataFrames
    merged_rain = merge_rain_gages(rain_dfs, site_ids)

    # Merge with stream DataFrame
    combined = stream_df.merge(merged_rain, on="collect_date", how="inner")

    print(f"Combined data shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")

    return combined


def preprocess_data(csv_path: Path, model_dir: Path, train_split: float):

    # merge all synced CSVs into a single DataFrame
    df = merge_rain_and_stream()

    rain_columns = [col for col in df.columns if "rain" in col.lower()]
    
    # Create rolling sum columns for each rain sensor at each configured window
    # for rain_col in rain_columns:
    #     for window in ROLLING_WINDOWS:
    #         rolling_sum_col = f"{rain_col}_rolling_sum_{window}"
    #         df[rolling_sum_col] = df[rain_col].rolling(window=window, min_periods=1).sum()
    
    # print(f"Added {len(rain_columns) * len(ROLLING_WINDOWS)} rolling sum features")
    print(f"Updated columns: {list(df.columns)}")
    
    # Save the enhanced combined data
    paired_dir = Path(__file__).parent.parent / 'data' / 'paired'
    df.to_csv(paired_dir / "combined_data.csv", index=False)

    # Drop timestamp if present
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # -------------------------------------------------
    # 2. Separate Inputs (Rain Sensors) and Output
    # -------------------------------------------------

    # Now use all rain columns plus the rolling sum features as inputs
    rain_columns = [col for col in df.columns if "rain" in col.lower()]
    target_column = "Stage"

    X = df[rain_columns].values
    y = df[target_column].values.reshape(-1, 1)

    # -------------------------------------------------
    # 3. Scale Data (IMPORTANT for NARX stability)
    # -------------------------------------------------

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(X)
    y_scaler.fit(y)
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y)

    # -------------------------------------------------
    # 4. Train/Test Split (Time-Aware Split Recommended)
    # -------------------------------------------------

    split_index = int(len(X_scaled) * train_split)

    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]

    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index:]

    # Save scalers
    with open(model_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)

    with open(model_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler

def build_narx_arrays(X, y, y_lag, x_lag):
    """Create NARX arrays from raw timeseries data.

    X: (N, n_sensors)
    y: (N,)

    Returns:
        X_narx: (N - lag, y_lag + x_lag * n_sensors) numpy array
        y_narx: (N - lag,) numpy array
    """
    N, n_sensors = X.shape
    X_narx = []
    y_narx = []

    for t in range(max(y_lag, x_lag), N):
        y_features = y[t - y_lag:t].flatten()       # past storm drain values
        x_features = X[t - x_lag:t].flatten()  # past rain sensor values
        X_narx.append(np.concatenate([y_features, x_features]))
        y_narx.append(y[t])

    X_narx = np.array(X_narx, dtype=np.float32)
    y_narx = np.array(y_narx, dtype=np.float32)
    return X_narx, y_narx


def create_narx_dataset(X, y) -> DataLoader:
    """Convenience wrapper that returns a DataLoader and the raw X_narx array."""
    X_narx, y_narx = build_narx_arrays(X, y, y_lag=Y_LAG, x_lag=X_LAG)

    # Convert to PyTorch
    X_tensor = torch.from_numpy(X_narx)
    y_tensor = torch.from_numpy(y_narx)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader, X_narx

def train_nn_narx(loader: DataLoader, X_narx: np.ndarray):
    """Train a PyTorch NARX network on the provided data loader.

    Returns:
        model (nn.Module): the trained NeuralNARX instance
    """
    input_size = X_narx.shape[1]
    model = NeuralNARX(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = nn.MSELoss()(y_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / len(yb)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.9f}")
        if epoch % 100 == 0 or epoch == epochs - 1:
            # Save model checkpoint every 10 epochs
            model_dir = Path(__file__).parent.parent / "saved_models"
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"narx_nn_epoch_{epoch+1}.pth")

    return model


# -------------------------------------------------
# Neural NARX evaluation helpers
# -------------------------------------------------

def _evaluate_nn(model, X_narx: np.ndarray, y_narx: np.ndarray, y_scaler, title: str, save_path=None):
    """Internal helper to run forward pass and plot results."""
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_narx)
        preds = model(inputs).cpu().numpy()

    # inverse scale
    y_pred = y_scaler.inverse_transform(preds)
    y_true = y_scaler.inverse_transform(y_narx.reshape(-1, 1))

    mse = mean_squared_error(y_true, y_pred)
    print(f"{title} MSE: {mse}")

    # plt.figure()
    # plt.plot(y_true, label="Actual")
    # plt.plot(y_pred, label="Predicted")
    # plt.legend()
    # plt.title(title)
    # plt.grid(True)
    # if save_path:
    #     plt.savefig(save_path, dpi=150)
    # plt.close()
    return y_pred, y_true, mse


def evaluate_nn_test(model, X_train, y_train, X_test, y_test, y_scaler):
    """Run the neural NARX model on the test set, prepending history from the train set."""
    lag = max(Y_LAG, X_LAG)
    # prepend last lag rows from training
    X_full = np.vstack([X_train[-lag:], X_test])
    y_full = np.vstack([y_train[-lag:], y_test])

    X_narx, y_narx = build_narx_arrays(X_full, y_full, y_lag=Y_LAG, x_lag=X_LAG)

    model_dir = Path(__file__).parent.parent / "saved_models"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "narx_nn_test_predictions.png"

    return _evaluate_nn(model, X_narx, y_narx, y_scaler, save_path,
                        "NNARX Prediction on Test Set")


def evaluate_nn_training(model, X_train, y_train, y_scaler, y_lag=Y_LAG, x_lag=X_LAG):
    """Run the neural NARX model on its training data for sanity-check."""
    X_narx, y_narx = build_narx_arrays(X_train, y_train, y_lag=Y_LAG, x_lag=X_LAG)

    model_dir = Path(__file__).parent.parent / "saved_models"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "narx_nn_train_predictions.png"

    return _evaluate_nn(model, X_narx, y_narx, y_scaler, save_path,
                        "NNARX Fit on Training Data")

def test_forecast(X_recent: list[pd.Dataframe], y_recent: pd.DataFrame, config: Config):
    """Use the trained NARX model to forecast future values given recent history."""

    # load model and scalers
    model = NeuralNARX(input_size=len(X_recent) * config.X_lag + config.y_lag)
    model.load_state_dict(torch.load(config.model_dir / config.model_name))

    with open(config.model_dir / config.x_scaler_name, "rb") as f:
        x_scaler: MinMaxScaler = pickle.load(f)   
    with open(config.model_dir / config.y_scaler_name, "rb") as f:
        y_scaler: MinMaxScaler = pickle.load(f)
    
    # first sync values
    stream_out_df, rain_out_dfs = process_gage_data(rain_dfs=X_recent, stream_df=y_recent[0])

    df = merge_rain_and_stream(rain_out_dfs, [f"site_{i}" for i in range(len(rain_out_dfs))], stream_out_df)

    rain_columns = [col for col in df.columns if "rain" in col.lower()]
    target_column = "stage"
    X = df[rain_columns].values
    y = df[target_column].values.reshape(-1, 1)

    X = x_scaler.transform(X)
    y = y_scaler.transform(y)

    # then build NARX input arrays
    X_narx, y_narx = build_narx_arrays(X, y, y_lag=config.y_lag, x_lag=config.X_lag)

    #save_path = config.model_dir / "narx_nn_test_forecast.png"

    return _evaluate_nn(model, X_narx, y_narx, y_scaler,
                        "NNARX Forecast Test")


def main(csv_path: Path, train_split: float, model_dir: Path):
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = preprocess_data(csv_path, model_dir, train_split)
    data_loader, X_narx = create_narx_dataset(X_train, y_train)
    model = train_nn_narx(data_loader, X_narx)

    # you can invoke the evaluation helpers below if desired
    evaluate_nn_test(model, X_train, y_train, X_test, y_test, y_scaler)
    evaluate_nn_training(model, X_train, y_train, y_scaler)

if __name__ == "__main__":
    csv_path = Path(r"/app/data/paired/combined_data.csv")
    model_dir = Path(r"/app/saved_models")
    train_split = 0.9
    main(csv_path, train_split, model_dir)