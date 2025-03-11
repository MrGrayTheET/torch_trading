import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from torch import nn as nn
from torch.utils.data import DataLoader


class TradingModel(nn.Module):
    """Neural network model for trading decisions."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize the trading model.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
        """
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, input_size)
        self.tanh = nn.Tanh()  # Output between -1 and 1 for position sizing

    def forward(self, x):
        """Forward pass to predict position size."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take only the last time step
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)
        return self.tanh(x)  # Range -1 to 1 (short to long)


class SharpeLoss(nn.Module):
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles

    def forward(self, y_true, weights):
        """
        Compute negative Sharpe ratio as a loss function

        Args:
            y_true: Ground truth returns
            weights: Model predicted portfolio weights
        """
        captured_returns = weights * y_true
        mean_returns = torch.mean(captured_returns)

        # Calculate Sharpe ratio (annualized)
        # We use negative Sharpe since we want to minimize loss
        return -(
                mean_returns
                / torch.sqrt(
            torch.mean(torch.square(captured_returns))
            - torch.square(mean_returns)
            + 1e-9
        )
                * torch.sqrt(torch.tensor(252.0))
        )


class SharpeValidationCallback:
    def __init__(
            self,
            model,
            inputs,
            returns,
            time_indices,
            num_time,  # including a count for nulls which will be indexed as 0
            early_stopping_patience,
            dataloader_kwargs=None,
            weights_save_location="tmp/checkpoint.pt",
            min_delta=1e-4,
    ):
        self.model = model
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta
        self.best_sharpe = float('-inf')  # since calculating positive Sharpe...
        self.weights_save_location = weights_save_location
        self.dataloader_kwargs = dataloader_kwargs or {}

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location

    def on_train_begin(self):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = float('-inf')

    def on_epoch_end(self, epoch):
        # Set model to evaluation mode
        self.model.eval()

        # Create a DataLoader for efficient prediction
        dataloader = DataLoader(
            self.inputs,
            **self.dataloader_kwargs
        )

        all_positions = []
        with torch.no_grad():
            for batch in dataloader:
                positions = self.model(batch)
                all_positions.append(positions)

        # Concatenate all predictions
        positions = torch.cat(all_positions, dim=0)

        # Convert to same device as returns if needed
        if positions.device != self.returns.device:
            positions = positions.to(self.returns.device)

        # Calculate time-segmented returns (similar to tf.math.unsorted_segment_mean)
        captured_returns = positions * self.returns

        # Group by time indices and calculate mean for each time segment
        segmented_returns = []
        for t in range(1, self.num_time):  # Skip index 0 (null times)
            mask = self.time_indices == t
            if torch.any(mask):
                segment_mean = torch.mean(captured_returns[mask])
                segmented_returns.append(segment_mean)

        captured_returns = torch.stack(segmented_returns)

        # Calculate Sharpe ratio
        sharpe = (
                torch.mean(captured_returns)
                / torch.sqrt(
            torch.var(captured_returns, unbiased=False)
            + torch.tensor(1e-9, dtype=torch.float64, device=captured_returns.device)
        )
                * torch.sqrt(torch.tensor(252.0, dtype=torch.float64, device=captured_returns.device))
        ).item()

        # Early stopping logic
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0  # reset the count
            torch.save(self.model.state_dict(), self.weights_save_location)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.load_state_dict(torch.load(self.weights_save_location))
                return True  # Signal to stop training

        print(f"\nval_sharpe {sharpe}")
        return False  # Continue training
