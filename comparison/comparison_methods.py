import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys

# Add the parent directory to the path so we can import from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import baseline random search
from main_lab3 import random_search

# Import our improved Bayesian method
from bayesian_optimization import bayesian_optimization_search

def compare_methods(file_path, budget, output_dir, repetitions=5):
    """
    Compare the performance of Random Search and Bayesian Optimization
    on a given dataset with multiple repetitions.
    
    Parameters:
        file_path (str): Path to the dataset CSV file
        budget (int): Number of iterations/configurations to try
        output_dir (str): Directory to save results
        repetitions (int): Number of times to repeat each method
    
    Returns:
        dict: Performance metrics for both methods
    """
    dataset_name = os.path.basename(file_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if maximization or minimization
    is_maximization = dataset_name.lower() == "---"  # No maximization in current dataset
    
    # Results containers
    random_results = []
    bayesian_results = []
    random_times = []
    bayesian_times = []
    
    # Run methods multiple times
    for i in range(repetitions):
        # Random Search
        rs_output = os.path.join(output_dir, f"{dataset_name}_random_{i}.csv")
        start_time = time.time()
        _, rs_performance = random_search(file_path, budget, rs_output)
        end_time = time.time()
        random_results.append(rs_performance)
        random_times.append(end_time - start_time)
        
        # Bayesian Optimization
        bo_output = os.path.join(output_dir, f"{dataset_name}_bayesian_{i}.csv")
        start_time = time.time()
        _, bo_performance = bayesian_optimization_search(file_path, budget, bo_output)
        end_time = time.time()
        bayesian_results.append(bo_performance)
        bayesian_times.append(end_time - start_time)
    
    # Process results based on problem type
    if is_maximization:
        random_best = max(random_results)
        bayesian_best = max(bayesian_results)
        improvement = ((bayesian_best - random_best) / abs(random_best)) * 100
    else:
        random_best = min(random_results)
        bayesian_best = min(bayesian_results)
        improvement = ((random_best - bayesian_best) / abs(random_best)) * 100
    
    # Calculate statistics
    metrics = {
        "dataset": dataset_name,
        "random_best": random_best,
        "bayesian_best": bayesian_best,
        "random_mean": np.mean(random_results),
        "bayesian_mean": np.mean(bayesian_results),
        "random_std": np.std(random_results),
        "bayesian_std": np.std(bayesian_results),
        "improvement_percent": improvement,
        "random_time_mean": np.mean(random_times),
        "bayesian_time_mean": np.mean(bayesian_times)
    }
    
    return metrics

def visualize_comparison(metrics, output_dir):
    """
    Create visualizations comparing Random Search and Bayesian Optimization.
    
    Parameters:
        metrics (list): List of metric dictionaries for each dataset
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plots
    datasets = [m["dataset"] for m in metrics]
    improvements = [m["improvement_percent"] for m in metrics]
    random_times = [m["random_time_mean"] for m in metrics]
    bayesian_times = [m["bayesian_time_mean"] for m in metrics]
    
    # 1. Bar chart of improvements
    plt.figure(figsize=(12, 6))
    bars = plt.bar(datasets, improvements, color='skyblue')
    for i, bar in enumerate(bars):
        if improvements[i] < 0:
            bar.set_color('salmon')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Improvement (%)', fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    plt.title('Performance Improvement of Bayesian Optimization over Random Search', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(improvements):
        plt.text(i, v + (5 if v >= 0 else -5), f"{v:.1f}%", ha='center', va='bottom' if v >= 0 else 'top')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_comparison.png'))
    
    # 2. Time comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.35
    plt.bar(x - width/2, random_times, width, label='Random Search', color='lightgreen')
    plt.bar(x + width/2, bayesian_times, width, label='Bayesian Optimization', color='lightblue')
    plt.ylabel('Average Execution Time (s)', fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    plt.title('Execution Time Comparison', fontsize=16)
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    
    # 3. Create a summary table
    summary_df = pd.DataFrame(metrics)
    summary_df.to_csv(os.path.join(output_dir, 'comparison_summary.csv'), index=False)
    
    # Display a table for console output
    print("\n===== Comparison Summary =====")
    print(f"{'Dataset':<15} {'Improvement':<15} {'Random Best':<15} {'Bayesian Best':<15}")
    print("-" * 60)
    for m in metrics:
        print(f"{m['dataset']:<15} {m['improvement_percent']:>6.2f}% {m['random_best']:>14.2f} {m['bayesian_best']:>14.2f}")

def parallel_compare(dataset_files, budget, output_dir, repetitions=5):
    """
    Run comparisons for multiple datasets in parallel.
    
    Parameters:
        dataset_files (list): List of dataset file paths
        budget (int): Number of iterations for each method
        output_dir (str): Directory to save results
        repetitions (int): Number of repetitions for each method
    """
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compare_methods, file_path, budget, output_dir, repetitions) 
                   for file_path in dataset_files]
        
        metrics = [future.result() for future in futures]
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Compare Configuration Tuning Methods')
    parser.add_argument('--budget', type=int, default=100, help='Number of iterations for each method')
    parser.add_argument('--repetitions', type=int, default=5, help='Number of repetitions for each method')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Get all dataset files
    datasets_folder = "datasets"
    dataset_files = [os.path.join(datasets_folder, f) for f in os.listdir(datasets_folder) if f.endswith('.csv')]
    
    # Run comparisons in parallel
    metrics = parallel_compare(dataset_files, args.budget, args.output_dir, args.repetitions)
    
    # Visualize results
    visualize_comparison(metrics, args.output_dir)

if __name__ == "__main__":
    main()