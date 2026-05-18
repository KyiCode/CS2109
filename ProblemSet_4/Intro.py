import torch
import torch.nn as nn
from torch.utils.data import DataLoader

### Task 1.1 - Define the model architecture and implement the forward pass
class DigitNet(nn.Module):
    def __init__(self, input_dimensions: int, num_classes: int): # set the arguments you'd need
        super().__init__()
        # - Create the 3 layers (and a ReLU layer) using the torch.nn layers API
        self.l1 = nn.Linear(input_dimensions, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return torch.softmax(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
        
def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 20):
    losses = []
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train() 
    for i in range(epochs):
        epoch_loss = 0.0
        samples_seen = 0
        for x_batch, y_batch in dataloader:
            """ YOUR CODE HERE """
            optimiser.zero_grad()
            outputs = model(x_batch)   
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimiser.step()
            """ YOUR CODE END HERE """
            epoch_loss += loss.item() * x_batch.shape[0]
            samples_seen += x_batch.shape[0]
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)

    return model, losses

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> int | float:
    correct = 0
    for i in range(scores.shape[0]):
        if torch.softmax(scores[i], dim=0).argmax() == labels[i]:
            correct += 1

    return correct / scores.shape[0] *100
