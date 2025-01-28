import torch

def save_model(model, path, save_state_dict=True):
    """
    Saves a PyTorch model to the specified path.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - path (str): The file path to save the model (e.g., 'model.pth').
    - save_state_dict (bool): If True, saves only the state_dict. If False, saves the entire model.

    Returns:
    None
    """
    if save_state_dict:
        # Save only the model's state_dict
        torch.save(model.state_dict(), path)
        print(f"Model's state_dict saved to {path}")
    else:
        # Save the entire model
        torch.save(model, path)
        print(f"Entire model saved to {path}")

def load_model(model, path, device='cpu'):
    """
    Loads a model's state_dict from a file.

    Parameters:
    - model (torch.nn.Module): The model architecture to load the state_dict into.
    - path (str): The file path to load the model state_dict from.
    - device (str): The device to map the model to ('cpu' or 'cuda').

    Returns:
    - model (torch.nn.Module): The model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model's state_dict loaded from {path}")
    return model