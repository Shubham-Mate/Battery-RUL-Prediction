import torch
import numpy as np
import matplotlib.pyplot as plt

def test_step(model, x, y, criterion, device, min, diff):
    """
    Perform a single testing step.

    Args:
        model (torch.nn.Module): The model to evaluate.
        x (torch.Tensor): Input data.
        y (torch.Tensor): Target data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the computations on (e.g., 'cpu' or 'cuda').
        min_val (float): Minimum value for normalization.
        diff_val (float): Difference (range) for normalization.

    Returns:
        float: The loss value for the step.
    """

    x = (x - min) / diff
    y = (y - min) / diff

    model.eval()

    with torch.no_grad():
        predictions = model(x.to(device))
        predictions = predictions.view(-1)

        loss = criterion(predictions, y.to(device))

    return loss.item()

def evaluate_model(model, dataloader, device, MIN, DIFF, unsqueeze_dim=1):
    """
    Evaluate a PyTorch model using RMSE and MAE on a given dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
    - MIN (float): Minimum value for normalization (if applicable).
    - DIFF (float): Difference value for normalization (if applicable).
    - unsqueeze_dim (int): Dimension to unsqueeze the input tensor (default is 1).

    Returns:
    - dict: A dictionary containing RMSE and MAE values.
    """
    # Define loss functions
    criterion_1 = torch.nn.MSELoss(reduction='sum')
    criterion_2 = torch.nn.L1Loss(reduction='sum')
    
    model.eval()
    total_samples = len(dataloader.dataset)
    test_loss_mse = 0.0
    test_loss_mae = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.unsqueeze(unsqueeze_dim)  # Apply unsqueeze with the given dimension
            test_loss_mse += test_step(model, x, y, criterion_1, device, MIN, DIFF)
            test_loss_mae += test_step(model, x, y, criterion_2, device, MIN, DIFF)

    avg_test_loss_mse = test_loss_mse / total_samples
    test_loss_rmse = np.sqrt(avg_test_loss_mse)
    avg_test_loss_mae = test_loss_mae / total_samples

    return {
        "RMSE": test_loss_rmse,
        "MAE": avg_test_loss_mae
    }

def plot_predictions(
    model, 
    dfs, 
    test_battery, 
    generate_sliding_window_data, 
    min_val, 
    diff_val, 
    device, 
    column_name="Discharge_Capacity(Ah)", 
    unsqueeze_dim=1, 
    cutoff_percentage=30,
    window_size=10
):
    """
    Generates predictions for the test curve using the given model and plots the results.

    Args:
        model (torch.nn.Module): Trained model for predictions.
        dfs (dict): Dictionary containing battery dataframes.
        test_battery (str): Key for the battery data in the dfs dictionary.
        generate_sliding_window_data (callable): Function to generate sliding window data.
        min_val (float): Minimum value used for normalization.
        diff_val (float): Range (difference) used for normalization.
        device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
        column_name (str, optional): Name of the column for discharge capacity. Defaults to "Discharge_Capacity(Ah)".
        unsqueeze_dim (int, optional): Dimension to unsqueeze before passing to the model. Defaults to 1.
        cutoff_percentage (int, optional): Percentage to determine the cutoff cycle. Defaults to 30.
    
    Returns:
        None
    """
    # Determine cutoff cycle based on percentage
    cutoff_cycle = int(cutoff_percentage * dfs[test_battery].shape[0] / 100)
    
    # Split initial and test curves
    initial = dfs[test_battery].loc[:cutoff_cycle, column_name].values
    test_curve = dfs[test_battery].loc[cutoff_cycle:, column_name].values

    # Generate sliding window data
    sliding_windows = generate_sliding_window_data(np.append(initial[-window_size:], test_curve), window_size)[0]

    model.eval()

    with torch.no_grad():
        # Normalize sliding window data
        sliding_windows_scaled = (sliding_windows - min_val) / diff_val

        # Generate predictions
        predictions = model(torch.tensor(sliding_windows_scaled).unsqueeze(unsqueeze_dim).to(device))
        predictions = predictions * diff_val + min_val
        predictions = predictions.flatten()

    # Cycle indices for plotting
    cycle_indices = np.arange(1, initial.shape[0] + test_curve.shape[0] + 1)
    cycle_indices_before = cycle_indices[:cutoff_cycle + 1]
    cycle_indices_after = cycle_indices[cutoff_cycle + 1:]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_indices_before, initial[:cutoff_cycle + 1], label="Initial Data", color="blue")
    plt.plot(cycle_indices_after, test_curve, label="True Test Curve", color="green")
    plt.plot(cycle_indices_after, predictions.cpu(), label="Predictions", color="red", linestyle="--")
    plt.xlabel("Cycle Index")
    plt.ylabel(column_name)
    plt.title(f"Predicted vs Actual {column_name}")
    plt.legend()
    plt.grid()
    plt.show()