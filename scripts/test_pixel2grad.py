import argparse
import numpy as np
import pytact
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Trains a pixel->grad MLP model using dataset from the create_grad_dataset script."
)
parser.add_argument("input_path", type=str, help="Path to test dataset")
parser.add_argument(
    "model_path",
    type=str,
    help="Path to saved model",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cuda", "cpu"],
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--batch_size", type=int, dest="batch_size", default=64)
args = parser.parse_args()

# Global parameters
device = torch.device(args.device)


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


model_path = args.model_path
model = pytact.models.Pixel2GradModel(
    hidden_size=96, dropout_p=0.1, activation=nn.ReLU(inplace=True)
)
model.load_state_dict(torch.load(model_path))
model.to(device)
# Create dataset
class Pixel2GradDataset(Dataset):
    def __init__(self, csv_file, to_gpu=True):
        self.labels = pd.read_csv(csv_file)
        self.to_gpu = to_gpu
        self.X = self.labels.iloc[:, 1:6].to_numpy().astype(np.float32)
        self.y = self.labels.iloc[:, 6:].to_numpy().astype(np.float32)
        print(
            "Dataset min/max values:",
            self.X.min(),
            self.X.max(),
            self.y.min(),
            self.y.max(),
        )
        ind = np.any(np.abs(self.y) > 2.0, axis=1)
        print(np.sum(ind), "outliers found in X")
        self.X = self.X[~ind, :]
        self.y = self.y[~ind, :]
        print(
            "Dataset min/max values after removing outliers:",
            self.X.min(),
            self.X.max(),
            self.y.min(),
            self.y.max(),
        )
        if self.to_gpu:
            self.X = torch.from_numpy(self.X).to(device)
            self.y = torch.from_numpy(self.y).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :]
        y = self.y[idx, :]
        return X, y


dataset = Pixel2GradDataset(args.input_path)

test_size = len(dataset)
test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

# Initiate model and optimizer

loss_fn = nn.MSELoss()
test(test_dataloader, model, loss_fn)
print("Done!")
