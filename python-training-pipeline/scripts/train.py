import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.generate_data import generate_data
from models.model import MyModel

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    # Generate synthetic data
    data = generate_data(num_samples=1000)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = MyModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

if __name__ == "__main__":
    main()