try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


def initialize_model(model_class, input_dim, output_dim, **kwargs):
    return model_class(input_dim, output_dim, **kwargs)


def save_model(model, path):
    if torch is None:
        raise ImportError("torch is required to save a model")
    torch.save(model.state_dict(), path)


def load_model(model_class, path, input_dim, output_dim, **kwargs):
    if torch is None:
        raise ImportError("torch is required to load a model")
    model = model_class(input_dim, output_dim, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def setup_optimizer(model, lr=0.001, optimizer_type="adam"):
    if optim is None:
        raise ImportError("torch is required to set up an optimizer")
    if optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def setup_loss_fn(loss_type="cross_entropy"):
    if nn is None:
        raise ImportError("torch is required to set up a loss function")
    if loss_type.lower() == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type.lower() == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
