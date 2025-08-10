class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_loss(self, predictions, targets):
        criterion = torch.nn.MSELoss()
        return criterion(predictions, targets)