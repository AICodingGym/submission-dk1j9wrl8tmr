"""Tapered BiLSTM with masked inspiratory scoring and ID-safe output."""

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "ventilator-pressure-prediction"
FEATURES = [
    "u_in", "u_out", "R", "C", "time_step",
    "dt", "dt_u_in", "volume", "u_in_lag1", "u_in_lead1",
    "u_out_lag1", "u_out_lead1", "u_in_cumsum", "u_in_roll5", "u_in_roll15",
    "R_times_C", "R_times_u_in", "volume_over_C", "R_over_C",
    "R_5", "R_20", "R_50", "C_10", "C_20", "C_50",
]
BATCH_SIZE = 256
EPOCHS = 30
ENSEMBLE_SEEDS = (42,)
EARLY_STOPPING_PATIENCE = 6


def add_features(frame):
    """Build breath-local history and lung-mechanics features."""
    frame = frame.sort_values(["breath_id", "time_step"]).copy()
    groups = frame.groupby("breath_id", sort=False)
    frame["dt"] = groups["time_step"].diff().fillna(0.0)
    frame["dt_u_in"] = frame["dt"] * frame["u_in"]
    frame["volume"] = frame.groupby("breath_id", sort=False)["dt_u_in"].cumsum()
    for column in ("u_in", "u_out"):
        frame[f"{column}_lag1"] = groups[column].shift(1).fillna(0.0)
        frame[f"{column}_lead1"] = groups[column].shift(-1).fillna(0.0)
    frame["u_in_cumsum"] = groups["u_in"].cumsum()
    frame["u_in_roll5"] = groups["u_in"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    frame["u_in_roll15"] = groups["u_in"].transform(
        lambda s: s.rolling(15, min_periods=1).mean()
    )
    frame["R_times_C"] = frame["R"] * frame["C"]
    frame["R_times_u_in"] = frame["R"] * frame["u_in"]
    frame["volume_over_C"] = frame["u_in_cumsum"] / frame["C"].clip(lower=1e-6)
    frame["R_over_C"] = frame["R"] / frame["C"].clip(lower=1e-6)
    for value in (5, 20, 50):
        frame[f"R_{value}"] = (frame["R"] == value).astype(np.float32)
    for value in (10, 20, 50):
        frame[f"C_{value}"] = (frame["C"] == value).astype(np.float32)
    return frame


def match_pid_breath(breath, pressure_levels):
    """Return exact-grid PID pressure matches for one breath, if available."""
    breath = breath.sort_values("time_step")
    inspiratory_length = int((breath["u_out"].to_numpy() == 0).sum())
    if inspiratory_length < 3:
        return {}

    # The simple proportional-controller relation is only evaluated after the
    # first point, and excludes valve endpoints that are not informative.
    candidate_rows = breath.iloc[1:inspiratory_length]
    u_in = candidate_rows["u_in"].to_numpy(dtype=np.float64)
    informative = (u_in > 0.0) & (u_in < 100.0)
    if informative.sum() < 2:
        return {}

    step = float(np.median(np.diff(pressure_levels)))
    minimum = float(pressure_levels[0])
    max_level = len(pressure_levels) - 1
    kp_grid = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
               2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    for kt in (10, 15, 20, 25, 30, 35):
        for kp in kp_grid:
            predictions = kt - u_in / kp
            grid_position = (predictions - minimum) / step
            rounded = np.rint(grid_position)
            if np.all((grid_position[informative] >= 0) &
                      (grid_position[informative] <= max_level) &
                      (np.abs(grid_position[informative] - rounded[informative]) < 1e-6)):
                row_ids = candidate_rows["id"].to_numpy(dtype=np.int64)
                return {
                    int(row_ids[index]): float(predictions[index])
                    for index in np.flatnonzero(informative)
                }
    return {}


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
            targets = np.zeros((length, 3), dtype=np.float32)
            targets[:, 0] = y
            if training:
                targets[1:, 1] = np.diff(y)
                targets[2:, 2] = y[2:] - 2.0 * y[1:-1] + y[:-2]
            # Keep the scoring mask and valid sequence length separate.
            score_mask = (breath["u_out"].to_numpy() == 0).astype(np.float32)
            ids = breath["id"].to_numpy(dtype=np.int64)
            self.samples.append((x, targets, score_mask, length, ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, mask, length, ids = self.samples[index]
        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy()),
            torch.from_numpy(mask.copy()),
            length,
            torch.from_numpy(ids.copy()),
        )


def collate_breaths(batch):
    lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    batch_size = len(batch)
    x = torch.zeros(batch_size, max_length, len(FEATURES), dtype=torch.float32)
    y = torch.zeros(batch_size, max_length, 3, dtype=torch.float32)
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
        self.front_end = nn.Sequential(
            nn.Conv1d(len(FEATURES), 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.residual_ffn = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64)
        )
        hidden_sizes = (256, 128)
        self.lstm_layers = nn.ModuleList()
        input_size = 64
        for hidden_size in hidden_sizes:
            self.lstm_layers.append(nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                batch_first=True, bidirectional=True,
            ))
            input_size = 2 * hidden_size
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Sequential(
            nn.Linear(input_size, 64), nn.SiLU(), nn.Linear(64, 3)
        )

    def forward(self, x, lengths):
        x = self.front_end(x.transpose(1, 2)).transpose(1, 2)
        steps = torch.arange(x.shape[1], device=x.device)[None, :]
        padding_mask = steps >= lengths.to(x.device)[:, None]
        transformed = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x + transformed
        x = x + self.residual_ffn(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out = packed
        for index, layer in enumerate(self.lstm_layers):
            packed_out, _ = layer(packed_out)
            if index + 1 < len(self.lstm_layers):
                packed_out = packed_out._replace(
                    data=self.dropout(packed_out.data)
                )
        sequence, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1]
        )
        return self.head(sequence)


def masked_mae(prediction, target, score_mask, lengths):
    prediction, target = prediction[..., 0], target[..., 0]
    steps = torch.arange(prediction.shape[1], device=prediction.device)[None, :]
    valid_mask = steps < lengths.to(prediction.device)[:, None]
    mask = score_mask * valid_mask.to(score_mask.dtype)
    loss_sum = (torch.abs(prediction - target) * mask).sum()
    return loss_sum / mask.sum().clamp_min(1.0), loss_sum.detach(), mask.sum().detach()


def masked_mse(prediction, target, score_mask, lengths):
    prediction, target = prediction[..., 0], target[..., 0]
    steps = torch.arange(prediction.shape[1], device=prediction.device)[None, :]
    valid_mask = steps < lengths.to(prediction.device)[:, None]
    mask = score_mask * valid_mask.to(score_mask.dtype)
    error_sum = (((prediction - target) ** 2) * mask).sum()
    return error_sum / mask.sum().clamp_min(1.0), error_sum.detach(), mask.sum().detach()


def run_epoch(model, loader, optimizer, device=None, objective="mae"):
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
            loss_fn = masked_mae if objective == "mae" else masked_mse
            loss, error_sum, count = loss_fn(prediction, y, mask, lengths)
            if training:
                velocity_loss, _, _ = masked_mae(
                    prediction[..., 1:2], y[..., 1:2], mask, lengths
                )
                acceleration_loss, _, _ = masked_mae(
                    prediction[..., 2:3], y[..., 2:3], mask, lengths
                )
                loss = loss + 0.25 * velocity_loss + 0.1 * acceleration_loss
                loss.backward()
                optimizer.step()
            total_error += error_sum.item()
            total_count += count.item()
    return total_error / max(total_count, 1.0)


def main():
    train = add_features(pd.read_csv(DATA_DIR / "train.csv"))
    test = add_features(pd.read_csv(DATA_DIR / "test.csv"))

    scaler = RobustScaler()
    scaler.fit(train[FEATURES])
    train_ds = BreathDataset(train, scaler, training=True)
    test_ds = BreathDataset(test, scaler, training=False)

    # Keep whole breaths together and balance the R/C settings in the holdout.
    breath_settings = train.groupby("breath_id", sort=False)[["R", "C"]].first()
    strata = breath_settings["R"].astype(str) + "_" + breath_settings["C"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_indices, val_indices = next(
        splitter.split(np.zeros(len(train_ds)), strata.to_numpy())
    )
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
    # Train two independently initialized models on the same breath holdout.
    # Keep each seed's best validation checkpoint and average them at inference.
    models = []
    for seed in ENSEMBLE_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PressureLSTM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )
        best_validation_mae = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(EPOCHS):
            run_epoch(model, train_loader, optimizer, device, objective="mae")
            val_mae = run_epoch(model, val_loader, optimizer=None, device=device)
            val_mse = run_epoch(
                model, val_loader, optimizer=None, device=device, objective="mse"
            )
            scheduler.step(val_mae)
            if val_mae < best_validation_mae:
                best_validation_mae = val_mae
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(
                f"Seed {seed}, epoch {epoch + 1}/{EPOCHS} - "
                f"Validation MAE: {val_mae:.5f} - Validation MSE: {val_mse:.5f} - "
                f"lr: {optimizer.param_groups[0]['lr']:.2g}",
                flush=True,
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Stopping early after {epoch + 1} epochs; validation MAE "
                    f"has not improved for {EARLY_STOPPING_PATIENCE} epochs.",
                    flush=True,
                )
                break
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        print(
            f"Seed {seed}: best validation MAE {best_validation_mae:.5f} "
            f"at epoch {best_epoch}.", flush=True,
        )

    # Critical: associate every prediction with its own id, never with the
    # original dataframe's positional order after grouping breaths.
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_breaths,
    )
    test_u_out = test.set_index("id")["u_out"].to_dict()
    predictions_by_id = {}
    with torch.no_grad():
        for x, _, _, lengths, ids in test_loader:
            x = x.to(device)
            predictions = np.median(
                [model(x, lengths).cpu().numpy()[..., 0] for model in models], axis=0
            )
            for i, length in enumerate(lengths.tolist()):
                for t in range(length):
                    row_id = int(ids[i, t])
                    if int(test_u_out[row_id]) == 0:
                        predictions_by_id[row_id] = float(predictions[i, t])

    # Replace model outputs only where the simple PID relation matches the
    # simulator's discrete pressure grid exactly; otherwise retain the model.
    pressure_levels = np.sort(train["pressure"].unique()).copy()
    pid_match_count = 0
    for _, breath in test.groupby("breath_id", sort=False):
        matches = match_pid_breath(breath, pressure_levels)
        predictions_by_id.update(matches)
        pid_match_count += len(matches)
    print(f"Applied exact PID matches to {pid_match_count:,} inspiratory rows.")

    submission = test[["id", "u_out"]].copy()
    prediction = submission["id"].map(predictions_by_id).fillna(0.0).to_numpy(copy=True)
    inspiratory = submission["u_out"].to_numpy() == 0
    positions = np.searchsorted(pressure_levels, prediction[inspiratory])
    lower = np.clip(positions - 1, 0, len(pressure_levels) - 1)
    upper = np.clip(positions, 0, len(pressure_levels) - 1)
    choose_upper = np.abs(pressure_levels[upper] - prediction[inspiratory]) < np.abs(
        pressure_levels[lower] - prediction[inspiratory]
    )
    snapped = np.where(choose_upper, pressure_levels[upper], pressure_levels[lower])
    prediction[inspiratory] = snapped
    submission["pressure"] = prediction
    submission[["id", "pressure"]].to_csv(ROOT / "submission.csv", index=False)
    print(f"Saved {len(submission):,} predictions to {ROOT / 'submission.csv'}")
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
