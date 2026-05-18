# RUN THIS CELL FIRST
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

import numpy as np
from numpy import allclose, isclose

from collections.abc import Callable
### Task 3.1 - Building RNN Model

class SineRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the SineRNN model.

        Args:
            input_size (int): The number of input features per time step (typically 1 for univariate time series).
            hidden_size (int): The number of units in the RNN's hidden layer.
            output_size (int): The size of the output (usually 1 for predicting a single value).
        """
        super(SineRNN, self).__init__()
        """ YOUR CODE HERE """
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        """ YOUR CODE END HERE """
        
    def forward(self, x):
        """ YOUR CODE HERE """
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        return output
        """ YOUR CODE END HERE """

# Define loss function, and optimizer
criterion = nn.MSELoss() # Quick quiz: Why are we using MSELoss?
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_seqs)
    loss = criterion(outputs.squeeze(), train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Predict on unseen data
model.eval()
y_pred = []
input_seq = test_seqs[0]  # Start with the first testing sequence

with torch.no_grad():
    for _ in range(len(test_seqs)):
        output = model(input_seq)
        y_pred.append(output.item())

        # Use the predicted value as the next input sequence
        next_seq = torch.cat((input_seq[1:, :], output.unsqueeze(0)), dim=0)
        input_seq = next_seq

# Plot the true sine wave and predictions
plt.plot(sine_wave_data, c='gray', label='Actual data')
plt.scatter(np.arange(seq_length + len(train_labels)), sine_wave_data[:seq_length + len(train_labels)], marker='.', label='Train')
x_axis_pred = np.arange(len(sine_wave_data) - len(test_labels), len(sine_wave_data))
plt.scatter(x_axis_pred, y_pred, marker='.', label='Predicted')
plt.legend(loc="lower left")
plt.show()

def create_sequences_with_noise(sine_wave, sine_wave_length, noise_length):
    """
    Create overlapping sequences from the input time series and generate labels.
    Each label is the value immediately following the corresponding sequence.
    Additionally, noise of the specified length is appended to the sequences.

    Args:
        sine_wave: A 1D tensor representing the time series data (e.g., sine wave).
        sine_wave_length: int. The length of the sine wave window.
        noise_length: int. The length of noise to be appended to each sequence.

    Returns:
        windows: 2D tensor where each row is a sequence of length `sine_wave_length + noise_length`.
        labels: 1D tensor where each element is the next value following each window.
    """
    windows = sine_wave.unfold(0, sine_wave_length, 1)
    labels = sine_wave[sine_wave_length:]
    noise = torch.randn(windows.shape[0], noise_length)
    windows = torch.cat((windows, noise), dim=1)
    return windows[:-1], labels

# Create sequences and labels
sine_wave_length = 20
noise_length = 20
sequences_noisy, labels_noisy = create_sequences_with_noise(sine_wave_data, sine_wave_length, noise_length)
# Add extra dimension to match RNN input shape [batch_size, seq_length, num_features]
sequences_noisy = sequences_noisy.unsqueeze(-1)
sequences_noisy.shape

# Split the sequences into training data (first 50%) and test data (remaining 50%) 
train_size = int(len(sequences_noisy) * 0.5)
train_seqs_noisy, train_labels = sequences_noisy[:train_size], labels_noisy[:train_size]
test_seqs_noisy, test_labels = sequences_noisy[train_size:], labels_noisy[train_size:]

# Define model
input_size = output_size = 1
hidden_size = 50
model = SineRNN(input_size, hidden_size, output_size).to(device)

# Define loss function, and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_seqs_noisy)
    loss = criterion(outputs.squeeze(), train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    y_pred = model(test_seqs_noisy).squeeze()
    y_true = test_labels.squeeze()

print("Test loss:", criterion(y_pred, y_true))

plt.figure(figsize=(8, 4))
plt.plot(y_true[1::2].numpy(), label="True value", color='black')
plt.plot(y_pred[1::2].numpy(), '--', label="Predicted value", color='red')
plt.title("SineRNN Predictions")
plt.xlabel("Test sequence index")
plt.ylabel("Target value")
plt.legend()
plt.show()

### Task 4.1 - Positional Encoding Layer

class PositionalEncoding(nn.Module):
    def __init__(self):
        # You do not need to change anything in this function.
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        You should use vectorized operations to compute the positional encoding.
        The use of Python loops is not allowed.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_4_1():
    encoder = PositionalEncoding()
    x0 = torch.zeros((1, 2, 4))
    y0 = encoder(x0)
    a0 = torch.tensor([[[0.0000, 1.0000, 0.0000, 1.0000],
                        [0.8415, 0.5403, 0.0100, 0.9999]]])
    
    assert isinstance(y0, torch.Tensor), "Output is not a tensor!"
    assert torch.allclose(y0, a0, atol=1e-4)
    
    x1 = torch.ones((1, 4, 6))
    y1 = encoder(x1)
    a1 = torch.tensor([[[1.0000, 2.0000, 1.0000, 2.0000, 1.0000, 2.0000],
                        [1.8415, 1.5403, 1.0464, 1.9989, 1.0022, 2.0000],
                        [1.9093, 0.5839, 1.0927, 1.9957, 1.0043, 2.0000],
                        [1.1411, 0.0100, 1.1388, 1.9903, 1.0065, 2.0000]]])
    
    assert torch.allclose(y1, a1, atol=1e-4)
    assert isinstance(y1, torch.Tensor), "Output is not a tensor!"

class TransformerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the TransformerNN model. We use the same hidden size for the feedforward network and the Transformer encoder.

        Args:
            input_size (int): The number of input features per time step (typically 1 for univariate time series).
            hidden_size (int): The number of units in the Transformer's hidden layers.
            output_size (int): The size of the output (usually 1 for predicting a single value).
        """
        super(TransformerNN, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoder = PositionalEncoding()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=hidden_size, nhead=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)

        # The encoder outputs a sequence of hidden states, so
        # we take the mean across the sequence length dimension.
        x = x.mean(dim=1)

        out = self.fc_out(x)
        return out

model = TransformerNN(input_size=1, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_seqs_noisy)
    loss = criterion(outputs.squeeze(), train_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')


model.eval()
with torch.no_grad():
    y_pred = model(test_seqs).squeeze()
    y_true = test_labels.squeeze()

print("Test loss:", criterion(y_pred, y_true))

plt.figure(figsize=(8, 4))
plt.plot(y_true[1::2].numpy(), label="True value", color='black')
plt.plot(y_pred[1::2].numpy(), '--', label="Predicted value", color='red')
plt.title("TransformerNN Predictions")
plt.xlabel("Test sequence index")
plt.ylabel("Target value")
plt.legend()
plt.show()

### Task 4.2 - Visualizing Attention Scores

### Task 4.3 - Discovering the Ingredients to Transformers' Success

# You may use this cell and create new cells to experiment.


if __name__ == '__main__':
    test_task_3_1()
    test_task_4_1()