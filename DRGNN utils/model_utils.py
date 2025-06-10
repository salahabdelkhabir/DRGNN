# utils/model_utils.py

import torch
import torch.nn as nn
import torch.optim as optim

# Function to initialize a model
def initialize_model(model_class, input_dim, output_dim, **kwargs):
    return model_class(input_dim, output_dim, **kwargs)

# Function to save a model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load a model
def load_model(model_class, path, input_dim, output_dim, **kwargs):
    model = model_class(input_dim, output_dim, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Function to set up an optimizer
def setup_optimizer(model, lr=0.001, optimizer_type="adam"):
    if optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

# Function to set up a loss function
def setup_loss_fn(loss_type="cross_entropy"):
    if loss_type.lower() == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type.lower() == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")

# Example usage
if __name__ == "__main__":
    class ExampleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(ExampleModel, self).__init__()
            self.layer = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.layer(x)

    # Initialize model
    model = initialize_model(ExampleModel, input_dim=10, output_dim=2)

    # Setup optimizer and loss function
    optimizer = setup_optimizer(model, lr=0.001)
    loss_fn = setup_loss_fn("cross_entropy")

    # Save and load model
    save_model(model, "example_model.pth")
    loaded_model = load_model(ExampleModel, "example_model.pth", input_dim=10, output_dim=2)
