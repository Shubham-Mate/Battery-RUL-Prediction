import torch
import numpy as np


def train_step(model, x, y, optimizer, criterion, device, min, diff):
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The model to train.
        x (torch.Tensor): Input data.
        y (torch.Tensor): Target data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the computations on (e.g., 'cpu' or 'cuda').
        min_val (float): Minimum value for normalization.
        diff_val (float): Difference (range) for normalization.

    Returns:
        float: The loss value for the step.
    """
    x = (x - min) / diff
    y = (y - min) / diff

    optimizer.zero_grad()

    predictions = model(x.to(device))
    predictions = predictions.view(-1)

    loss = criterion(predictions, y.to(device))

    loss.backward()
    optimizer.step()

    return loss.item()

def train_model(
    model, 
    train_dataloader, 
    optimizer, 
    criterion, 
    device, 
    min_val, 
    diff_val, 
    epochs, 
    unsqueeze_dim=1, 
    log_interval=1
):
    """
    Trains a PyTorch model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (callable): Loss function.
        device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
        min_val (float): Minimum value for normalization.
        diff_val (float): Range (difference) used for normalization.
        epochs (int): Number of training epochs.
        unsqueeze_dim (int, optional): Dimension to unsqueeze input data. Defaults to 1.
        log_interval (int, optional): Interval for logging loss. Defaults to 1.

    Returns:
        None
    """
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for x, y in train_dataloader:
            # Perform a training step
            loss_item = train_step(model, x.unsqueeze(unsqueeze_dim), y, optimizer, criterion, device, min_val, diff_val)
            epoch_loss += loss_item

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_dataloader)

        # Log progress at specified intervals
        if epoch % log_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")