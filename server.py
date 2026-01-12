import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

import torch
from src.model import FirePatchConvLSTM
from src.utils import predict_from_manual_input, get_anchor_dates
from src.config import config

app = FastAPI()

ckpt_path = "models/best.pt"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sample_input = {
    "output_date": 20180305,
    "patch_row": 80,
    "patch_col": 32,
}


class inputModel(BaseModel):
    output_date: int
    patch_row: int
    patch_col: int


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def run_single_inference_example(input_dict: inputModel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    in_channels = len(config["VARS"]) + (1 if config["ADD_MISSINGNESS_MASK"] else 0)
    model = FirePatchConvLSTM(
        in_channels=in_channels, hidden=64, lstm_layers=1, kernel=3, dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    eg = input_dict.model_dump()

    final_date = eg.pop("output_date")
    eg["year"] = int(datetime.strptime(str(final_date), "%Y%m%d").year)
    eg["t_steps"] = get_anchor_dates(final_date)

    result = predict_from_manual_input(model, eg, config, device=device)

    return {"result": result}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
