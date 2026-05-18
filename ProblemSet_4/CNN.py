import torch
import torch.nn as nn
from torchvision import datasets, transforms

### Task 2.1 - Building a CNN that uses Conv2d, MaxPool2d, and Dropout layers.
class DropoutCNN(nn.Module):
    def __init__(self, classes: int, drop_prob: float = 0.5):
        super().__init__()
        # classes: integer that corresponds to the number of classes for MNIST
        # drop_prob: probability of dropping a node in the neural network
        self.l1 = nn.Conv2d(1, 32, (3,3))
        self.mp1 = nn.MaxPool2d((2,2))
        self.l2 = nn.LeakyReLU(0.1)
        self.do1 = nn.Dropout(drop_prob)  # dropout
        self.l3 = nn.Conv2d(32, 64, (3,3))
        self.mp2 = nn.MaxPool2d((2,2))
        self.l4 = nn.LeakyReLU(0.1)
        self.do2 = nn.Dropout(drop_prob)  # dropout
        # flatten
        self.l6 = nn.Linear(1600, 256)
        self.l7 = nn.LeakyReLU(0.1)
        self.do3 = nn.Dropout(drop_prob)
        self.l8 = nn.Linear(256, 128)
        self.l9 = nn.LeakyReLU(0.1)
        self.l10 = nn.Linear(128, classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.mp1(x)
        x = self.l2(x)
        x = self.do1(x)
        x = self.l3(x)
        x = self.mp2(x)
        x = self.l4(x)
        x = self.do2(x)
        
        x = x.view(-1, 64*5*5) # Flattening – do not remove

        x = self.l6(x)
        x = self.l7(x)
        x = self.do3(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return torch.softmax(out, dim = 1)

### Task 2.3 - Picking Data Augmentations
def get_augmentations() -> transforms.Compose:
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.ColorJitter(brightness=0.2), 
    ])
    return T

### Task 2.4 - Write a Custom Data Augmentation
class SimulateRGBA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        # Adds a new channel to `image` that simulates an alpha channel.
        # image: a tensor of shape (C, H, W) 
        #     C is the number of channels
        #     H is the height
        #     W is the width of the image
        #     values in the range [0, 1]
        a = image.mean(dim = 0, keepdim = True)
        return [image, a]

def get_augmentations() -> transforms.Compose:
    T = transforms.Compose([
        transforms.ToTensor(),
        # ShuffleChannels(), # Uncomment this line to test out the demo ShuffleChannels transformation
        SimulateRGBA(), 
        # --- You are free to experiment with more custom transformations! ---
        # Add in your data augmentations from Task 2.3 here
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip()
    ])
    return T

T = get_augmentations()

### Task 2.5 - Build a ConvNet for CIFAR-10
class CIFARCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        # classes: integer that corresponds to the number of classes for CIFAR-10
        self.conv = nn.Sequential(
                        nn.Conv2d(4, 32, (3,3)),  # 4 channels due to rgba
                        nn.MaxPool2d(2,2),
                        nn.LeakyRelu(0.1),
                        nn.conv2d(64, (3,3)),
                        nn.MaxPool2d(2,2),
                        nn.LeakyRelu(0.1)
                    )
        self.fc = nn.Sequential(
                        nn.Linear(64,256),
                        nn.LeakyRelu(0.1),
                        nn.Linear(256, 128),
                        nn.LeakyRelu(0.1),
                        nn.Linear(128, self.classes)
                    )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x= self.conv(x)
        x = x.view(x.shape[0], 64, 6*6).mean(2) # GAP – do not remove this line
        x = self.fc(x)
        out = x
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return torch.softmax(out, dim = 1)