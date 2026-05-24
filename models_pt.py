import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import mean_squared_error

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ANN(nn.Module):
    def __init__(self, seq_length, n_features):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(seq_length * n_features, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_features)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        return self.fc3(x)

class CNN(nn.Module):
    def __init__(self, seq_length, n_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1D(n_features, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPooling1D(2)
        self.conv2 = nn.Conv1D(16, 32, kernel_size=3, padding=1)
        conv_output_size = seq_length // 2
        self.fc = nn.Linear(32 * conv_output_size, n_features)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class RNN(nn.Module):
    def __init__(self, input_size, n_features):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, 32, batch_first=True)
        self.fc = nn.Linear(32, n_features)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n[-1])

def train_evaluate_pt(model, train_dataset, test_loader, batch_size=16, epochs=10):
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            y_pred.append(output.cpu().numpy())
            y_true.append(y_batch.numpy())
    
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return rmse, y_pred, train_losses, val_losses
