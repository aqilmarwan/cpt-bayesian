import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import time

# Function to calculate expected improvement
def expected_improvement(X, model, y_best, xi=0.01):
    """
    Calculate expected improvement for given points using a trained model.
    
    Parameters:
        X (array): Points to evaluate EI for
        model (GaussianProcessRegressor): Trained GP model
        y_best (float): Best observed value
        xi (float): Exploration-exploitation parameter
    
    Returns:
        array: Expected improvement values
    """
    mu, sigma = model.predict(X, return_std=True)
    
    # For minimization problems, use negative values
    if not is_maximization_problem:
        mu = -mu
        y_best = -y_best
    
    # Calculate improvement
    with np.errstate(divide='warn'):
        imp = mu - y_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-8] = 0.0
    
    # Return negative values for minimization problems
    if not is_maximization_problem:
        return -ei
    return ei

# Enhanced bayesian optimization search function
def bayesian_optimization_search(file_path, budget, output_file):
    """
    Perform Bayesian Optimization to find optimal configuration.
    
    Parameters:
        file_path (str): Path to CSV file with configuration data
        budget (int): Number of iterations/configurations to try
        output_file (str): Path to save search results
    
    Returns:
        list: Best configuration found
        float: Performance of the best configuration
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Identify columns for configurations and performance
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    
    # Determine if this is a maximization or minimization problem
    system_name = os.path.basename(file_path).split('.')[0]
    global is_maximization_problem
    if system_name.lower() == "---":  # No maximization problems in current dataset
        is_maximization_problem = True
    else:
        is_maximization_problem = False
    
    # Extract the best and worst performance values
    if is_maximization_problem:
        worst_value = data[performance_column].min() / 2  # For missing configurations
        best_value = -np.inf
        direction = 1  # For sorting (higher is better)
    else:
        worst_value = data[performance_column].max() * 2  # For missing configurations
        best_value = np.inf
        direction = -1  # For sorting (lower is better)
    
    # Initialize the best solution and performance
    best_performance = best_value
    best_solution = []
    
    # Store all search results
    search_results = []
    
    # Create a copy of the dataset for tracking explored configurations
    explored_data = data.copy()
    
    # Create a list of all possible configuration values for each parameter
    config_values = {}
    for col in config_columns:
        config_values[col] = sorted(data[col].unique())
    
    # Setup for GP model
    # Initialize with a few random samples (10% of budget or at least 5)
    init_samples = max(int(budget * 0.1), 5)
    X_sample = []
    y_sample = []
    
    # Random initial sampling
    for i in range(init_samples):
        # Sample a random configuration
        sampled_config = [int(np.random.choice(config_values[col])) for col in config_columns]
        
        # Check if configuration exists in dataset
        matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]
        
        if not matched_row.empty:
            performance = matched_row[performance_column].iloc[0]
        else:
            performance = worst_value
        
        # Add to samples
        X_sample.append(sampled_config)
        y_val = performance
        if not is_maximization_problem:
            y_val = -y_val  # Convert to maximization problem for GP
        y_sample.append(y_val)
        
        # Update best solution
        if (is_maximization_problem and performance > best_performance) or \
           (not is_maximization_problem and performance < best_performance):
            best_performance = performance
            best_solution = sampled_config
        
        # Record the search result
        search_results.append(sampled_config + [performance])
    
    # Convert samples to numpy arrays
    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)
    
    # For remaining budget, use Bayesian Optimization
    for i in range(init_samples, budget):
        # Train a Gaussian Process model
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42
        )
        
        # Fit the model on all observed configurations
        gp.fit(X_sample, y_sample)
        
        # Use Thompson Sampling with exploration-exploitation balance
        # First generate candidate configurations
        n_candidates = 1000
        candidate_configs = []
        
        # Mix of random sampling (exploration) and neighborhood sampling (exploitation)
        if len(X_sample) > 0 and i % 5 != 0:  # 80% exploitation
            # Sample around best configuration with some noise
            for _ in range(n_candidates):
                base_config = X_sample[np.argmax(y_sample)]
                candidate = []
                for j, col in enumerate(config_columns):
                    # 70% chance to mutate each parameter
                    if np.random.random() < 0.7:
                        valid_values = config_values[col]
                        # Choose a neighboring value with higher probability
                        current_idx = valid_values.index(base_config[j])
                        if len(valid_values) > 1:
                            # Sample with weight decaying with distance from current value
                            weights = np.exp(-0.5 * np.abs(np.arange(len(valid_values)) - current_idx))
                            weights = weights / np.sum(weights)
                            new_idx = np.random.choice(len(valid_values), p=weights)
                            candidate.append(valid_values[new_idx])
                        else:
                            candidate.append(valid_values[0])
                    else:
                        candidate.append(base_config[j])
                candidate_configs.append(candidate)
        else:  # 20% exploration - completely random
            for _ in range(n_candidates):
                candidate_configs.append([int(np.random.choice(config_values[col])) for col in config_columns])
        
        # Convert candidates to numpy array
        candidate_configs = np.array(candidate_configs)
        
        # Predict performance with uncertainties for all candidates
        mu, sigma = gp.predict(candidate_configs, return_std=True)
        
        # Sample from these predictions (Thompson Sampling)
        thompson_samples = np.random.normal(mu, sigma)
        
        # Select best candidate according to Thompson Sampling
        best_idx = np.argmax(thompson_samples)
        next_config = candidate_configs[best_idx].tolist()
        
        # Ensure we're using integers for configuration parameters
        next_config = [int(val) for val in next_config]
        
        # Check if configuration exists in dataset
        matched_row = data.loc[(data[config_columns] == pd.Series(next_config, index=config_columns)).all(axis=1)]
        
        if not matched_row.empty:
            performance = matched_row[performance_column].iloc[0]
        else:
            performance = worst_value
        
        # Add to samples
        X_sample = np.vstack([X_sample, next_config])
        y_val = performance
        if not is_maximization_problem:
            y_val = -y_val  # Convert to maximization problem for GP
        y_sample = np.append(y_sample, y_val)
        
        # Update best solution
        if (is_maximization_problem and performance > best_performance) or \
           (not is_maximization_problem and performance < best_performance):
            best_performance = performance
            best_solution = next_config
        
        # Record the search result
        search_results.append(next_config + [performance])
    
    # Save the search results to a CSV file
    columns = list(config_columns) + ["Performance"]
    search_df = pd.DataFrame(search_results, columns=columns)
    search_df.to_csv(output_file, index=False)
    
    return [int(x) for x in best_solution], best_performance

# Import for EI calculation
from scipy.stats import norm

# Main function
def main():
    # Set parameters
    datasets_folder = "datasets"
    output_folder = "bo_search_results"
    visualization_folder = "bo_visualization_results"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(visualization_folder, exist_ok=True)
    budget = 100  # Same as original for fair comparison
    
    # Tracking metrics for comparison
    results = {}
    
    # Process each dataset
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            print(f"Processing {file_name}...")
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # Run Bayesian Optimization search
            start_time = time.time()
            best_solution, best_performance = bayesian_optimization_search(file_path, budget, output_file)
            end_time = time.time()
            
            # Store results
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Search Time": end_time - start_time
            }
            
            # Create visualization
            visualize_search_results(output_folder, file_name.split('.')[0], visualization_folder)
    
    # Print the results
    print("\n===== Search Results =====")
    for system, result in results.items():
        print(f"System: {system}")
        print(f"  Best Solution:    [{', '.join(map(str, result['Best Solution']))}]")
        print(f"  Best Performance: {result['Best Performance']}")
        print(f"  Search Time:      {result['Search Time']:.2f} seconds")
        print()

# Visualization function
def visualize_search_results(results_folder, dataset_name, visualization_folder):
    """
    Visualize the search results and compare with baseline if available.
    
    Parameters:
        results_folder (str): Folder containing the search results CSV files
        dataset_name (str): Name of the dataset to visualize (without extension)
        visualization_folder (str): Folder to save the visualization images
    """
    # Construct the file paths
    csv_file = os.path.join(results_folder, f"{dataset_name}_search_results.csv")
    output_image = os.path.join(visualization_folder, f"{dataset_name}_visualization.png")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The results file {csv_file} does not exist.")
        return
    
    # Load the search results
    search_df = pd.read_csv(csv_file)
    
    # Determine if this is a maximization or minimization problem
    is_maximization = dataset_name.lower() == "---"  # No maximization in current dataset
    
    # Find the best performance value and its index
    if is_maximization:
        best_performance = search_df["Performance"].max()
    else:
        best_performance = search_df["Performance"].min()
    
    best_index = search_df[search_df["Performance"] == best_performance].index[0]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Calculate running best performance
    if is_maximization:
        running_best = search_df["Performance"].expanding().max()
    else:
        running_best = search_df["Performance"].expanding().min()
    
    # Plot the performance values
    plt.plot(search_df.index, search_df["Performance"], marker="o", linestyle="-", 
             alpha=0.5, label="Performance")
    
    # Plot the running best performance
    plt.plot(search_df.index, running_best, 'g-', linewidth=2, 
             label="Running Best Performance")
    
    # Highlight the best point
    plt.plot(best_index, best_performance, marker="*", color="red", markersize=15, 
             label="Best Point")
    
    # Try to load baseline results for comparison
    baseline_folder = "search_results"
    baseline_file = os.path.join(baseline_folder, f"{dataset_name}_search_results.csv")
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        if is_maximization:
            baseline_best = baseline_df["Performance"].max()
        else:
            baseline_best = baseline_df["Performance"].min()
        
        # Add baseline best as horizontal line
        plt.axhline(y=baseline_best, color='r', linestyle='--', alpha=0.7,
                    label=f"Baseline Best: {baseline_best:.2f}")
        
        # Add improvement text
        improvement = ((baseline_best - best_performance) / baseline_best * 100)
        if not is_maximization:
            improvement = -improvement  # Flip sign for minimization
        
        plt.text(0.02, 0.02, f"Improvement: {improvement:.2f}%", 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add labels and title
    plt.xlabel("Search Iteration", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.title(f"Bayesian Optimization Results for {dataset_name}", fontsize=16)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Save and close the plot
    os.makedirs(visualization_folder, exist_ok=True)
    plt.savefig(output_image)
    plt.close()

if __name__ == "__main__":
    main()