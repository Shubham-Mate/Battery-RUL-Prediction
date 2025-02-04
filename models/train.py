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

def validation_step(model, x, y, criterion, device, min_val, diff_val):
    """
    Perform a single validation step.

    Args:
        model (torch.nn.Module): The model to evaluate.
        x (torch.Tensor): Input data.
        y (torch.Tensor): Target data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run computations on (e.g., 'cpu' or 'cuda').
        min_val (float): Minimum value for normalization.
        diff_val (float): Difference (range) for normalization.

    Returns:
        float: The loss value for the step.
    """
    x = (x - min_val) / diff_val
    y = (y - min_val) / diff_val

    with torch.no_grad():
        predictions = model(x.to(device))
        predictions = predictions.view(-1)

        loss = criterion(predictions, y.to(device))

    return loss.item()


def train_model(
    model, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    criterion, 
    device, 
    min_val, 
    diff_val, 
    epochs, 
    unsqueeze_dim=1, 
    log_interval=1,
    patience=5, 
    save_path="best_model.pth"
):
    """
    Trains a PyTorch model with validation and early stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (callable): Loss function.
        device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
        min_val (float): Minimum value for normalization.
        diff_val (float): Range (difference) used for normalization.
        epochs (int): Number of training epochs.
        unsqueeze_dim (int, optional): Dimension to unsqueeze input data. Defaults to 1.
        log_interval (int, optional): Interval for logging loss. Defaults to 1.
        patience (int, optional): Number of epochs with no improvement before early stopping. Defaults to 5.
        save_path (str, optional): Path to save the best model. Defaults to "best_model.pth".

    Returns:
        None
    """
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for x, y in train_dataloader:
            loss_item = train_step(model, x.unsqueeze(unsqueeze_dim), y, optimizer, criterion, device, min_val, diff_val)
            train_loss += loss_item

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                loss_item = validation_step(model, x_val.unsqueeze(unsqueeze_dim), y_val, criterion, device, min_val, diff_val)
                val_loss += loss_item

        avg_val_loss = val_loss / len(val_dataloader)

        # Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), save_path) 
            print(f"Epoch {epoch + 1}: Validation loss improved to {avg_val_loss}. Model saved!")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch + 1}: No improvement. ({epochs_no_improve}/{patience})")

        # EarlyStopping with Patience
        if epochs_no_improve >= patience:
            print("Early stopping triggered. Training stopped.")
            break

        if epoch % log_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")