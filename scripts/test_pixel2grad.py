import argparse

import pytact
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pixel2grad import Pixel2GradDataset, test

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

model_path = args.model_path
model = pytact.models.Pixel2GradModel()
model.load_state_dict(torch.load(model_path))
model.to(device)

dataset = Pixel2GradDataset(args.input_path, device)

test_size = len(dataset)
test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

# Initiate model and optimizer

loss_fn = nn.MSELoss()
test(test_dataloader, model, loss_fn, device)
print("Done!")
