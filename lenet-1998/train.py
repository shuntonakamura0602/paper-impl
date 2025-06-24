
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LeNet5

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False
    )

    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    print(f"Test Accuracy: {correct / len(test_loader.dataset):.4f}")

if __name__ == "__main__":
    train()
