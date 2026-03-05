import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

from app.model import StreamflowNet
from pathlib import Path
import pandas as pd

def get_training_data():
    """
    Load synced CSVs from data/paired directory and merge into a single DataFrame.
    Features come from each rain site and target is the stream stage.
    """
    paired_dir = Path(__file__).parent.parent / 'data' / 'paired'
    # find stream file
    stream_files = list(paired_dir.glob('stream-gages_*_synced_smoothed.csv'))
    if not stream_files:
        raise RuntimeError(f"No synced stream file in {paired_dir}")
    stream_df = pd.read_csv(stream_files[0], parse_dates=['Date'])
    stream_df = stream_df.set_index('Date')

    # load rain files and join
    rain_files = list(paired_dir.glob('rain-gages_*_synced_smoothed.csv'))
    if not rain_files:
        raise RuntimeError(f"No synced rain files in {paired_dir}")
    merged = stream_df.copy()
    for rf in rain_files:
        df = pd.read_csv(rf, parse_dates=['Date']).set_index('Date')
        # rename rain column to unique name based on filename
        colname = rf.stem.replace('_synced','')
        df = df.rename(columns={'Rain Amount (in)': colname})
        merged = merged.join(df, how='inner')
    merged = merged.dropna()
    # reset index to include Date if needed
    merged = merged.reset_index()
    return merged


def preprocess(df):
    """
    Separate features (rain amounts) from target (Stage) and return numpy arrays.
    """
    X = df.drop(columns=['Date', 'Stage']).values
    y = df['Stage'].values
    return X, y


def train_model():
    df = get_training_data()
    X, y = preprocess(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = StreamflowNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    epochs = 200

    for epoch in range(epochs):
        # training step
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(loader.dataset)
        train_losses.append(epoch_loss)

        # validation
        model.eval()
        with torch.no_grad():
            Xv = torch.FloatTensor(X_val)
            yv = torch.FloatTensor(y_val).unsqueeze(1)
            pred_val = model(Xv)
            vloss = loss_fn(pred_val, yv).item()
            val_losses.append(vloss)

        print(f"Epoch {epoch+1}/{epochs}, train loss {epoch_loss:.6f}, val loss {vloss:.6f}")

        # checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            saved_models_dir = Path(__file__).parent.parent / "saved_models"
            saved_models_dir.mkdir(exist_ok=True)
            checkpoint_path = saved_models_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint {checkpoint_path}")

    # final save
    saved_models_dir = Path(__file__).parent.parent / "saved_models"
    saved_models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), saved_models_dir / "model.pt")
    joblib.dump(scaler, saved_models_dir / "scaler.save")

    # plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), train_losses, label='train')
    plt.plot(range(1, epochs+1), val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_plot_path = saved_models_dir / 'loss_curve.png'
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")
    plt.close()

if __name__ == "__main__":
    train_model()