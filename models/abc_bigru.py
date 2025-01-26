import torch
import numpy as np

class BiGRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.bigru = torch.nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x, _ = self.bigru(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Define the evaluation function for the ABC algorithm
def evaluate_hyperparameters(params, train_dataloader, val_dataloader, device):
    hidden_size, learning_rate, dropout = params
    hidden_size = int(hidden_size)

    # Initialize the model, loss, and optimizer
    model = BiGRUModel(input_size=10, hidden_size=hidden_size, dropout=dropout).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(3):  # Small number of epochs for fast evaluation
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X.unsqueeze(1).to(device))
            loss = criterion(outputs.squeeze(), batch_y.to(device))
            loss.backward()
            optimizer.step()

    # Validate the model
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            outputs = model(batch_X.unsqueeze(1).to(device)).squeeze()
            loss = criterion(outputs, batch_y.to(device))
            total_loss += loss.item()

    return total_loss / len(val_dataloader)  # Return average MSE as fitness


# ABC algorithm
def modify_solution(solution, bounds, factor=0.1):
    new_solution = solution + np.random.uniform(-factor, factor, size=len(solution))
    return np.clip(new_solution, bounds[:, 0], bounds[:, 1])

def abc_algorithm(bounds, train_dataloader, val_dataloader, device, n_solutions=10, max_iters=20):
    n_params = bounds.shape[0]
    solutions = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_solutions, n_params))
    fitness = np.array([evaluate_hyperparameters(s, train_dataloader, val_dataloader, device) for s in solutions])

    best_solution = solutions[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(max_iters):
        for i in range(n_solutions):
            # Employed bee phase
            new_solution = modify_solution(solutions[i], bounds)
            new_fitness = evaluate_hyperparameters(new_solution, train_dataloader, val_dataloader, device)
            if new_fitness < fitness[i]:
                solutions[i] = new_solution
                fitness[i] = new_fitness

        # Onlooker bee phase
        probabilities = fitness / np.sum(fitness)
        for i in range(n_solutions):
            if np.random.rand() < probabilities[i]:
                new_solution = modify_solution(solutions[i], bounds)
                new_fitness = evaluate_hyperparameters(new_solution, train_dataloader, val_dataloader, device)
                if new_fitness < fitness[i]:
                    solutions[i] = new_solution
                    fitness[i] = new_fitness

        # Scout bee phase
        if np.random.rand() < 0.1:  # 10% chance to explore randomly
            random_index = np.random.randint(n_solutions)
            solutions[random_index] = np.random.uniform(bounds[:, 0], bounds[:, 1], size=n_params)
            fitness[random_index] = evaluate_hyperparameters(solutions[random_index], train_dataloader, val_dataloader, device)

        # Update the best solution
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness:
            best_solution = solutions[current_best_index]
            best_fitness = fitness[current_best_index]

        print(f"Iteration {iteration + 1}/{max_iters}, Best Fitness (MSE): {best_fitness}")

    return best_solution, best_fitness

