"""Two-layer LSTM baseline with masked inspiratory MAE and ID-safe output."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "ventilator-pressure-prediction"
FEATURES = ["u_in", "u_out", "R", "C", "time_step"]
BATCH_SIZE = 256
EPOCHS = 5


class BreathDataset(Dataset):
    def __init__(self, frame, scaler, training):
        self.samples = []
        for _, breath in frame.groupby("breath_id", sort=False):
            breath = breath.sort_values("time_step")
            x = scaler.transform(breath[FEATURES]).astype(np.float32)
            length = len(breath)
            y = (
                breath["pressure"].to_numpy(dtype=np.float32)
                if training
                else np.zeros(length, dtype=np.float32)
            )
            # Keep the scoring mask and valid sequence length separate.
            score_mask = (breath["u_out"].to_numpy() == 0).astype(np.float32)
            ids = breath["id"].to_numpy(dtype=np.int64)
            self.samples.append((x, y, score_mask, length, ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, mask, length, ids = self.samples[index]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), length, torch.from_numpy(ids)


def collate_breaths(batch):
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
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(FEATURES), hidden_size=128, num_layers=2, batch_first=True
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        sequence, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1]
        )
        return self.output(sequence).squeeze(-1)


def masked_mae(prediction, target, score_mask, lengths):
    steps = torch.arange(prediction.shape[1], device=prediction.device)[None, :]
    valid_mask = steps < lengths.to(prediction.device)[:, None]
    mask = score_mask * valid_mask.to(score_mask.dtype)
    loss_sum = (torch.abs(prediction - target) * mask).sum()
    return loss_sum / mask.sum().clamp_min(1.0), loss_sum.detach(), mask.sum().detach()


def run_epoch(model, loader, optimizer, device=None):
    training = optimizer is not None
    model.train(training)
    total_error = 0.0
    total_count = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, mask, lengths, _ in loader:
            if training:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
            else:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
            prediction = model(x, lengths)
            loss, error_sum, count = masked_mae(prediction, y, mask, lengths)
            if training:
                loss.backward()
                optimizer.step()
            total_error += error_sum.item()
            total_count += count.item()
    return total_error / max(total_count, 1.0)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    scaler = StandardScaler()
    scaler.fit(train[FEATURES])
    train_ds = BreathDataset(train, scaler, training=True)
    test_ds = BreathDataset(test, scaler, training=False)

    # Hold out complete breaths for a trustworthy validation MAE.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(train_ds))
    split = int(0.9 * len(order))
    train_indices, val_indices = order[:split].tolist(), order[split:].tolist()
    train_loader = DataLoader(
        Subset(train_ds, train_indices), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, collate_fn=collate_breaths,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_indices), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, collate_fn=collate_breaths,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Training on {device}; input features: {FEATURES}")
    model = PressureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        run_epoch(model, train_loader, optimizer, device)
        val_mae = run_epoch(model, val_loader, optimizer=None, device=device)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Validation MAE: {val_mae:.5f}", flush=True)

    # Refit for the same number of epochs on every labeled breath.
    full_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_breaths,
    )
    model = PressureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        run_epoch(model, full_loader, optimizer, device)

    # Critical: associate every prediction with its own id, never with the
    # original dataframe's positional order after grouping breaths.
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_breaths,
    )
    test_u_out = test.set_index("id")["u_out"].to_dict()
    predictions_by_id = {}
    model.eval()
    with torch.no_grad():
        for x, _, _, lengths, ids in test_loader:
            predictions = model(x.to(device), lengths).cpu().numpy()
            for i, length in enumerate(lengths.tolist()):
                for t in range(length):
                    row_id = int(ids[i, t])
                    if int(test_u_out[row_id]) == 0:
                        predictions_by_id[row_id] = float(predictions[i, t])

    submission = test[["id", "u_out"]].copy()
    submission["pressure"] = submission["id"].map(predictions_by_id).fillna(0.0)
    submission[["id", "pressure"]].to_csv(ROOT / "submission.csv", index=False)
    print(f"Saved {len(submission):,} predictions to {ROOT / 'submission.csv'}")
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
