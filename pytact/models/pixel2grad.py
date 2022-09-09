import torch.nn as nn
from pytact.types import ModelType


class Pixel2GradModel(nn.Module):
    """
    A 3-layer MLP to convert visuo-tactile pixels into depth gradients.

    Architecture: 5 (R, G, B, x, y) -> 64 -> 64 -> 64 -> 2 (gx, gy)
    """

    model_type = ModelType.Pixel2Grad
    dropout_p = 0.05

    def __init__(self, hidden_size=64, dropout_p=0.05, activation=nn.ReLU(True)):
        super().__init__()
        self.activation = activation

        self.fc1 = nn.Linear(5, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.drop(x)
        x = self.activation(self.fc2(x))
        x = self.drop(x)
        x = self.activation(self.fc3(x))
        x = self.drop(x)
        return self.fc4(x)
