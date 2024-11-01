# neural_network.py
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def train_model(model, X_train, y_train, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        # Convert to tensor
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.long)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test):
    inputs = torch.tensor(X_test, dtype=torch.float32)
    labels = torch.tensor(y_test, dtype=torch.long)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    # Load data
    from build_dataset import load_data
    X_train, X_test, y_train, y_test = load_data("iris.csv")

    # Model parameters
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = len(set(y_train))

    # Initialize model, criterion and optimizer
    model = SimpleFeedForwardNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    train_model(model, X_train, y_train, criterion, optimizer, num_epochs=100)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
