"""Packed-sequence LSTM baseline for ventilator pressure prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "ventilator-pressure-prediction"
FEATURES = ["u_in", "u_out", "R", "C", "time_step"]
BATCH_SIZE = 256
EPOCHS = 5


class BreathDataset(Dataset):
    """Store each breath as a variable-length sequence."""

    def __init__(self, frame, feature_mean, feature_scale, training):
        self.breaths = []
        for _, breath in frame.groupby("breath_id", sort=False):
            breath = breath.sort_values("time_step")
            x = breath[FEATURES].to_numpy(dtype=np.float32)
            x = (x - feature_mean) / feature_scale
            length = len(breath)

            if training:
                y = breath["pressure"].to_numpy(dtype=np.float32)
                score_mask = (breath["u_out"].to_numpy() == 0).astype(np.float32)
                ids = np.full(length, -1, dtype=np.int64)
            else:
                y = np.zeros(length, dtype=np.float32)
                score_mask = np.zeros(length, dtype=np.float32)
                ids = breath["id"].to_numpy(dtype=np.int64)

            self.breaths.append((x, y, score_mask, length, ids))

    def __len__(self):
        return len(self.breaths)

    def __getitem__(self, index):
        x, y, score_mask, length, ids = self.breaths[index]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(score_mask),
            length,
            torch.from_numpy(ids),
        )


def collate_breaths(batch):
    """Pad each batch and return lengths for packed-sequence processing."""
    lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    batch_size = len(batch)
    x = torch.zeros(batch_size, max_length, len(FEATURES), dtype=torch.float32)
    y = torch.zeros(batch_size, max_length, dtype=torch.float32)
    score_mask = torch.zeros(batch_size, max_length, dtype=torch.float32)
    ids = torch.full((batch_size, max_length), -1, dtype=torch.long)

    for i, (seq_x, seq_y, seq_mask, length, seq_ids) in enumerate(batch):
        x[i, :length] = seq_x
        y[i, :length] = seq_y
        score_mask[i, :length] = seq_mask
        ids[i, :length] = seq_ids

    return x, y, score_mask, lengths, ids


class PressureLSTM(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        sequence, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )
        return self.output(sequence).squeeze(-1)


def masked_mae(prediction, target, score_mask, lengths):
    # The length mask excludes padding; score_mask excludes u_out == 1.
    steps = torch.arange(prediction.size(1), device=prediction.device)[None, :]
    valid_mask = steps < lengths.to(prediction.device)[:, None]
    mask = score_mask * valid_mask.to(score_mask.dtype)
    error = torch.abs(prediction - target) * mask
    return error.sum() / mask.sum().clamp_min(1.0), mask.sum()


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train = train.sort_values(["breath_id", "time_step"])
    test = test.sort_values(["breath_id", "time_step"])

    feature_mean = train[FEATURES].mean().to_numpy(dtype=np.float32)
    feature_scale = train[FEATURES].std().to_numpy(dtype=np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0

    dataset = BreathDataset(train, feature_mean, feature_scale, training=True)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(dataset))
    split = int(0.9 * len(order))
    train_set = Subset(dataset, order[:split].tolist())
    validation_set = Subset(dataset, order[split:].tolist())

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_breaths,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_breaths,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS is unavailable; using CPU instead.")
    print(f"Training on {device} with packed variable-length sequences.")

    model = PressureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for x, target, score_mask, lengths, _ in train_loader:
            x = x.to(device)
            target = target.to(device)
            score_mask = score_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, lengths)
            loss, _ = masked_mae(prediction, target, score_mask, lengths)
            loss.backward()
            optimizer.step()

        model.eval()
        total_error = 0.0
        total_scored = 0.0
        with torch.no_grad():
            for x, target, score_mask, lengths, _ in validation_loader:
                x = x.to(device)
                target = target.to(device)
                score_mask = score_mask.to(device)
                prediction = model(x, lengths)
                loss, count = masked_mae(prediction, target, score_mask, lengths)
                total_error += loss.item() * count.item()
                total_scored += count.item()
        val_mae = total_error / max(total_scored, 1.0)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Validation MAE: {val_mae:.5f}", flush=True)

    test_dataset = BreathDataset(test, feature_mean, feature_scale, training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_breaths,
    )
    model.eval()
    predictions = {}
    test_u_out = test.set_index("id")["u_out"].to_dict()
    with torch.no_grad():
        for x, _, _, lengths, ids in test_loader:
            batch_predictions = model(x.to(device), lengths).cpu().numpy()
            for i, length in enumerate(lengths.tolist()):
                for step in range(length):
                    row_id = int(ids[i, step])
                    if int(test_u_out[row_id]) == 0:
                        predictions[row_id] = float(batch_predictions[i, step])

    submission = test[["id", "u_out"]].copy()
    submission["pressure"] = submission["id"].map(predictions).fillna(0.0)
    submission = submission[["id", "pressure"]]
    output_path = ROOT / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
