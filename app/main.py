from fastapi import FastAPI
import torch
import joblib
import numpy as np

from model import StreamflowNet

app = FastAPI()

model = StreamflowNet(input_dim=2)
model.load_state_dict(torch.load("/app/saved_models/model.pt"))
model.eval()

scaler = joblib.load("/app/saved_models/scaler.save")


@app.post("/predict")
def predict(rain_1: float, rain_2: float):
    features = np.array([[rain_1, rain_2]])
    features = scaler.transform(features)

    with torch.no_grad():
        prediction = model(torch.FloatTensor(features))

    return {"predicted_streamflow": prediction.item()}