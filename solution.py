import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 1. Load Data
train = pd.read_csv('data/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('data/ventilator-pressure-prediction/test.csv')

features = ['u_in', 'u_out', 'R', 'C', 'time_step']
target = 'pressure'

# 2. Scale features 
scaler = StandardScaler()
train_feats = scaler.fit_transform(train[features])
test_feats = scaler.transform(test[features])

train[features] = train_feats
test[features] = test_feats

# 3. Create sequences group by breath_id
def create_sequences(df, is_train=True):
    X, y, u_out = [], [], []
    for b_id, group in df.groupby('breath_id'):
        X.append(group[features].values)
        u_out.append(group['u_out'].values)
        if is_train:
            y.append(group[target].values)
    if is_train:
        return X, y, u_out
    return X, u_out

print("Building train sequences...")
X_train, y_train, u_out_train = create_sequences(train, is_train=True)
print("Building test sequences...")
X_test, u_out_test = create_sequences(test, is_train=False)

class VentilatorDataset(Dataset):
    def __init__(self, X, y, u_out):
        self.X = X
        self.y = y
        self.u_out = u_out
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32), 
                torch.tensor(self.y[idx], dtype=torch.float32), 
                torch.tensor(self.u_out[idx], dtype=torch.float32))

train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 4. Model
class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LSTMNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.L1Loss(reduction='none')

# 5. Training Loop
print(f"Training on {device}...")
for epoch in range(5):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y, batch_u in train_loader:
        batch_X, batch_y, batch_u = batch_X.to(device), batch_y.to(device), batch_u.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_X)
        
        # ONLY mask where u_out == 0 (inspiratory phase)
        mask = (batch_u == 0)
        loss = loss_fn(preds[mask], batch_y[mask]).mean()
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/5 | Masked Train MAE: {epoch_loss/len(train_loader):.4f}")

# 6. Predict Test
print("Generating predictions...")
model.eval()
all_preds = []
with torch.no_grad():
    for i in range(len(X_test)):
        seq = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(seq).squeeze().cpu().numpy()
        all_preds.extend(pred)

# Create submission
sub = pd.DataFrame({'id': test['id'], 'pressure': all_preds})
# Force expiratory phase to 0 (since it's not scored and u_out==1 is just air leaving)
sub.loc[test['u_out'] == 1, 'pressure'] = 0

sub.to_csv('submission.csv', index=False)
print("Saved submission.csv successfully!")