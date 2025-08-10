import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
from torch import optim
from torch.utils.data import DataLoader
from data.generate_data import generate_data
from models.model import MyModel

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def main():
    data = generate_data(num_samples=1000)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    model = MyModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    dummy_input = torch.randn(1, data.num_features)
    onnx_path = "models/mlp.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=11)
    print(f"Modelo exportado para {onnx_path}")

if __name__ == "__main__":
    main()
