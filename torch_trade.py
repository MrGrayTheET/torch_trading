import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  datasets import  ModelDataset
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



# Define the CNN model
class CNNTimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim,input_len, output_size, kernel_size=3, stride=1, padding=0):
        super(CNNTimeSeriesPredictor, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_output_size = self._calculate_conv_output_size(input_len, kernel_size, stride, padding)
        self.fc1 = nn.Linear(128 * self.conv_output_size , 64)
        self.fc2 = nn.Linear(64, output_size)

    def _calculate_conv_output_size(self, input_len, kernel_size, stride, padding):
        # Calculate the output size after conv1d_1
        conv1_output = (input_len - kernel_size + 2 * padding) // stride + 1
        # Calculate the output size after maxpool
        maxpool_output = (conv1_output - 2) // 2 + 1
        # Calculate the output size after conv1d_2
        conv2_output = (maxpool_output - kernel_size + 2 * padding) // stride + 1
        return conv2_output

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, sequence_length, num_features, output_sequence_length,
                 kernel_sizes=[3, 5, 7], feature_maps=[64, 128, 256]):
        """
        CNN for time series analysis with multidimensional data

        Args:
            input_channels: Number of input channels (e.g., number of stocks or features per time step)
            sequence_length: Length of input time sequence
            num_features: Number of features per channel
            output_sequence_length: Length of the output sequence (future time steps to predict)
            kernel_sizes: List of kernel sizes for each convolutional layer
            feature_maps: List of feature maps for each convolutional layer
        """
        super(TimeSeriesCNN, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_sequence_length = output_sequence_length
        self.feature_extractors = nn.ModuleList()

        for _ in range(input_channels):
            layers = nn.ModuleList()

            # First conv layer
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=feature_maps[0],
                          kernel_size=kernel_sizes[0], padding='same'),
                nn.BatchNorm1d(feature_maps[0]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ))

            # Additional conv layers
            for i in range(1, len(feature_maps)):
                layers.append(nn.Sequential(
                    nn.Conv1d(in_channels=feature_maps[i - 1], out_channels=feature_maps[i],
                              kernel_size=kernel_sizes[min(i, len(kernel_sizes) - 1)], padding='same'),
                    nn.BatchNorm1d(feature_maps[i]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2) if i < len(feature_maps) - 1 else nn.Identity()
                ))

            self.feature_extractors.append(layers)
        # Calculate the size after convolutions and pooling
        conv_output_size = sequence_length // (2 ** (len(feature_maps) - 1))
        single_extractor_size = feature_maps[-1] * conv_output_size
        combined_features_size = single_extractor_size * input_channels


        # Attention mechanism for temporal relationships
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_maps[-1],
            num_heads=4,
            batch_first=True
        )

        # Fully connected layers for output prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_features_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_sequence_length)  # Single series output
        )

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape [batch_size, input_channels, sequence_length, num_features]

        Returns:
            Predicted returns of shape [batch_size, input_channels, output_sequence_length]
        """
        batch_size = x.shape[0]

        # Process each input channel separately
        extracted_features = []

        for channel_idx in range(self.input_channels):
            # Extract this channel's data: [batch_size, sequence_length, num_features]
            channel_data = x[:, channel_idx, :, :]

            # Reshape for 1D convolution: [batch_size, num_features, sequence_length]
            channel_data = channel_data.transpose(1, 2)

            # Pass through this channel's convolutional layers
            feature_maps = channel_data
            for conv_layer in self.feature_extractors[channel_idx]:
                feature_maps = conv_layer(feature_maps)

            # Flatten the output for this channel
            flattened = feature_maps.reshape(batch_size, -1)
            extracted_features.append(flattened)

        # Combine all extracted features
        combined = torch.cat(extracted_features, dim=1)

        # Apply cross-series attention (optional)
        # Reshape for attention if needed

        # Pass through fully connected layers to get final prediction
        output = self.fc_layers(combined)

        return output



# Example usage

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
        self.fc2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()  # Output between -1 and 1 for position sizing

    def forward(self, x):
        """Forward pass to predict position size."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take only the last time step
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)  # Range -1 to 1 (short to long)

        return x


def train_model(model: nn.Module, train_dataset: ModelDataset,
                val_dataset: ModelDataset, date_index, output_size: int = 1, num_epochs: int = 50,
                batch_size: int = 1, learning_rate: float = 0.001):
    """
    Train the trading model using a custom Sharpe ratio loss.
    
    Args:
        model: PyTorch model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Lists of training and validation Sharpe ratios per epoch
    """

    loader_kwargs = dict(batch_size=1, shuffle=False)
    output_size = len(train_dataset[1][1])
    input_len = len(train_dataset[0][0])

    train_loader = DataLoader(
        train_dataset, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, **loader_kwargs
    )
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Lists to track training and testing losses
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for features, returns in train_loader:
            # Skip batches with insufficient data
            if (returns.shape[1] < output_size) or (features.shape[1] < input_len):
                continue

            # Zero out the gradients before backward pass
            optimizer.zero_grad()

            # Reshape features correctly according to model's expected input
            # Assuming model expects [batch_size, channels, sequence_length, features]
            features = features.reshape(1, features.shape[0], features.shape[1], features.shape[2])

            # Forward pass
            prediction = model(features)

            # Calculate loss
            loss = criterion(prediction, returns)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            batch_count += 1

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_epoch_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_epoch_loss:.4f}")

        # Validation phase
        model.eval()
        test_loss = 0.0
        test_batch_count = 0

        with torch.no_grad():
            for features, returns in val_loader:
                if (returns.shape[1] != output_size) or (features.shape[1] < input_len):
                    continue

                features = features.reshape(1, features.shape[0],features.shape[1], features.shape[2])
                prediction = model(features)
                loss = criterion(prediction, returns)
                test_loss += loss.item()
                test_batch_count += 1

        avg_test_loss = test_loss / test_batch_count if test_batch_count > 0 else 0
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}")

    return  train_losses, test_losses

def plot_results(results, data):
    """Plot trading performance and positions."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # Plot portfolio value
    dates = data.index[-len(results['portfolio_values']):]
    ax1.plot(dates, results['portfolio_values'])
    ax1.set_title('Portfolio Value')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True)
    
    # Plot positions
    position_dates = dates[1:]
    ax2.fill_between(position_dates, results['positions'], alpha=0.5)
    ax2.set_title('Position Size (-1 to 1)')
    ax2.set_ylabel('Position')
    ax2.grid(True)
    
    # Plot stock price
    ax3.plot(data.index, data['close'])
    ax3.set_title('Stock Price')
    ax3.set_ylabel('Price ($)')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Create text summary
    metrics_text = (
        f"Sharpe Ratio: {results['sharpe_ratio']:.4f}\n"
        f"Final Balance: ${results['final_balance']:.2f}\n"
        f"Total Return: {(results['final_balance'] / 10000 - 1) * 100:.2f}%\n"
        f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%"
    )
    
    fig.text(0.15, 0.02, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    
    return fig


