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

from app.model import NeuralNARX

# Configurable rolling window sizes (in timesteps) for rolling sum features
ROLLING_WINDOWS = [3, 6, 12]  # Create rolling sums at these horizons

X_LAG = 10
Y_LAG = 10


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

    # -------------------------------------------------
    # 1.5. Create Rolling Sum Features
    # -------------------------------------------------

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

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

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

    # get correlation
    # for i in range(X_train.shape[1]):
    #     corr = np.correlate(X_train[:, i], y_train.flatten(), mode="full")
    #     lag = corr.argmax() - len(X_train)
    #     print(f"Max correlation for feature {i} at lag {lag} timesteps")

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler

def build_narx_arrays(X, y, y_lag=10, x_lag=10):
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

    for t in range(max(Y_LAG, X_LAG), N):
        y_features = y[t - Y_LAG:t].flatten()       # past storm drain values
        x_features = X[t - X_LAG:t].flatten()  # past rain sensor values
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
    y_tensor = torch.from_numpy(y_narx).unsqueeze(1)

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
            epoch_loss += loss.item() * len(yb)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.6f}")
        if epoch % 100 == 0 or epoch == epochs - 1:
            # Save model checkpoint every 10 epochs
            model_dir = Path(__file__).parent.parent / "saved_models"
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"narx_nn_epoch_{epoch+1}.pth")

    return model


def train_model(X_train, y_train, x_scaler, y_scaler):

    # -------------------------------------------------
    # 5. Define NARX Model
    # -------------------------------------------------

    model = FROLS(
        order_selection=True,
        n_info_values=15,       # model complexity cap
        ylag=20,                 # past 20 storm drain values
        xlag=[10, 10],   # past 10 rain values (each is 10 min apart)
        basis_function=Polynomial(degree=1),
        #estimator=LeastSquares(),      # IMPORTANT for stability
        estimator=RidgeRegression(alpha=0.01)
    )

    # -------------------------------------------------
    # 6. Train Model
    # -------------------------------------------------

    model.fit(X=X_train, y=y_train)
    print(model.final_model)

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
        y_scaler: MinMaxScaler = pickle.load(f)

    # For NARX with lag, we need to prepend the last lag values from train to test
    y_lag = 20  # ylag from model
    xlag = 10  # xlag from model
    lag = max(y_lag, xlag)
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


def evaluate_on_training(X_train, y_train):
    """Run the trained model on the training set and report/plot results.

    This is useful to verify the model is actually fitting the data rather than
    just producing flat predictions.
    """
    model_dir = Path(__file__).parent.parent / "saved_models"

    # load model and scalers
    with open(model_dir / "narx_model.pkl", "rb") as f:
        model: FROLS = pickle.load(f)
    with open(model_dir / "y_scaler.pkl", "rb") as f:
        y_scaler: MinMaxScaler = pickle.load(f)

    # prediction on training set (no need to prepend extra lag because y_train
    # already contains the necessary history for itself)
    y_pred_scaled = model.predict(X=X_train, y=y_train)

    # reverse scaling
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_actual = y_scaler.inverse_transform(y_train)

    mse = mean_squared_error(y_actual, y_pred)
    print(f"Training MSE: {mse}")

    # plot and save
    plt.figure()
    plt.plot(y_actual, label="Actual (train)")
    plt.plot(y_pred, label="Predicted (train)")
    plt.legend()
    plt.title("NARX Fit on Training Data")
    plt.grid(True)
    plt.savefig(model_dir / "narx_train_predictions.png", dpi=150)
    plt.close()


# -------------------------------------------------
# Neural NARX evaluation helpers
# -------------------------------------------------

def _evaluate_nn(model, X_narx: np.ndarray, y_narx: np.ndarray, y_scaler, save_path, title: str):
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

    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return mse


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


def main(csv_path: Path, train_split: float):
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = preprocess_data(csv_path, train_split)
    data_loader, X_narx = create_narx_dataset(X_train, y_train)
    model = train_nn_narx(data_loader, X_narx)

    # you can invoke the evaluation helpers below if desired
    evaluate_nn_test(model, X_train, y_train, X_test, y_test, y_scaler)
    evaluate_nn_training(model, X_train, y_train, y_scaler)

    # train_model(X_train, y_train, x_scaler, y_scaler)
    # evaluate_model(X_train, y_train, X_test, y_test)
    # evaluate_on_training(X_train, y_train)

if __name__ == "__main__":
    csv_path = Path("path_to_your_data.csv")
    train_split = 0.9
    main(csv_path, train_split)