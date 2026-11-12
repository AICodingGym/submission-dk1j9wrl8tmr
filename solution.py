"""Train an LSTM pressure predictor and write submission.csv.

Run from the repository root with: python solution.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).parent / "data" / "ventilator-pressure-prediction"
FEATURES = ["u_in", "u_out", "R", "C", "time_step"]
BATCH_SIZE = 256
EPOCHS = 5


class BreathDataset(Dataset):
    """One item per breath, with rows sorted in time order."""

    def __init__(self, frame, max_length, feature_mean, feature_scale, training):
        self.max_length = max_length
        self.training = training
        self.breaths = []
        for _, breath in frame.groupby("breath_id", sort=False):
            breath = breath.sort_values("time_step")
            x = breath[FEATURES].to_numpy(dtype=np.float32)
            x = (x - feature_mean) / feature_scale
            length = len(breath)
            if training:
                y = breath["pressure"].to_numpy(dtype=np.float32)
                score_mask = (breath["u_out"].to_numpy() == 0).astype(np.float32)
                row_ids = None
            else:
                y = np.zeros(length, dtype=np.float32)
                score_mask = np.zeros(length, dtype=np.float32)
                row_ids = breath["id"].to_numpy(dtype=np.int64)
            self.breaths.append((x, y, score_mask, length, row_ids))

    def __len__(self):
        return len(self.breaths)

    def __getitem__(self, index):
        x, y, score_mask, length, row_ids = self.breaths[index]
        pad = self.max_length - length
        x = np.pad(x, ((0, pad), (0, 0)))
        y = np.pad(y, (0, pad))
        score_mask = np.pad(score_mask, (0, pad))
        if row_ids is not None:
            row_ids = np.pad(row_ids, (0, pad), constant_values=-1)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(score_mask),
            length,
            torch.from_numpy(row_ids) if row_ids is not None else torch.empty(0),
        )


class PressureLSTM(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence).squeeze(-1)


def masked_mae(prediction, target, mask):
    absolute_error = torch.abs(prediction - target) * mask
    return absolute_error.sum() / mask.sum().clamp_min(1.0)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    max_length = max(
        int(train.groupby("breath_id").size().max()),
        int(test.groupby("breath_id").size().max()),
    )

    feature_mean = train[FEATURES].mean().to_numpy(dtype=np.float32)
    feature_scale = train[FEATURES].std().to_numpy(dtype=np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0

    train_dataset = BreathDataset(
        train, max_length, feature_mean, feature_scale, training=True
    )
    test_dataset = BreathDataset(
        test, max_length, feature_mean, feature_scale, training=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS is unavailable; using CPU instead.")
    print(f"Training on {device}; maximum breath length is {max_length}.")

    model = PressureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_error = 0.0
        total_scored = 0.0
        for x, target, mask, _, _ in train_loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = masked_mae(prediction, target, mask)
            loss.backward()
            optimizer.step()
            total_error += loss.item() * mask.sum().item()
            total_scored += mask.sum().item()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - inspiratory MAE: "
            f"{total_error / max(total_scored, 1):.5f}",
            flush=True,
        )

    model.eval()
    predictions_by_id = {}
    with torch.no_grad():
        for x, _, _, lengths, row_ids in test_loader:
            prediction = model(x.to(device)).cpu().numpy()
            row_ids = row_ids.numpy()
            for batch_index, length in enumerate(lengths.tolist()):
                for step in range(length):
                    predictions_by_id[int(row_ids[batch_index, step])] = float(
                        prediction[batch_index, step]
                    )

    submission = test[["id", "u_out"]].copy()
    submission["pressure"] = submission["id"].map(predictions_by_id)
    submission.loc[submission["u_out"] == 1, "pressure"] = 0.0
    submission = submission[["id", "pressure"]]
    output_path = DATA_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
