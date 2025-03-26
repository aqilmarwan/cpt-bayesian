import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

# Sample data - would be replaced with real data from experiments
# Iteration numbers
iterations = np.arange(100)

# Simulated data based on the Apache system results described in the report
# For Bayesian Optimization: starts higher, drops quickly, finds good solution by ~30 iterations
np.random.seed(42)  # For reproducibility
# Simulate Bayesian Optimization results (starts high, drops quickly, stabilizes at low value)
bo_mean = 340 * np.ones(100)
# Rapid improvement in first 30 iterations
for i in range(1, 30):
    bo_mean[i] = max(bo_mean[i-1] * 0.9 - np.random.exponential(10), 20)
# Slower refinement afterwards
for i in range(30, 100):
    bo_mean[i] = max(bo_mean[i-1] - np.random.exponential(0.5), 18.46)
    
# Add some noise/variation
bo_std = np.ones(100) * 5
bo_std[:10] = 20  # Higher variation at the beginning
bo_std[10:30] = 10  # Medium variation during rapid improvement

# Simulate Random Search results (more random exploration, slower convergence)
rs_mean = 340 * np.ones(100)
# Very occasional improvements
improvement_indices = [18, 20, 35, 38, 41, 45, 52, 60, 73, 82, 92]
for i in range(1, 100):
    if i in improvement_indices:
        # Big improvement
        rs_mean[i] = max(rs_mean[i-1] * 0.5, 30.8)
    elif np.random.random() < 0.1:
        # Small improvement
        rs_mean[i] = max(rs_mean[i-1] * 0.95, 30.8)
    else:
        # No improvement
        rs_mean[i] = rs_mean[i-1]

# Add variation
rs_std = np.ones(100) * 15

# Create running best arrays (non-increasing for minimization problem)
bo_running_best = np.minimum.accumulate(bo_mean)
rs_running_best = np.minimum.accumulate(rs_mean)

# Create the figure
plt.figure(figsize=(12, 8))

# Plot Bayesian Optimization with standard deviation band
plt.plot(iterations, bo_running_best, 'b-', linewidth=2.5, label='Bayesian Optimization')
plt.fill_between(iterations, 
                 bo_running_best - bo_std, 
                 bo_running_best + bo_std, 
                 color='blue', alpha=0.15)

# Plot Random Search with standard deviation band
plt.plot(iterations, rs_running_best, 'orange', linewidth=2.5, label='Random Search')
plt.fill_between(iterations, 
                 rs_running_best - rs_std, 
                 rs_running_best + rs_std, 
                 color='orange', alpha=0.15)

# Mark the best point found by Bayesian Optimization
best_bo_value = bo_running_best[-1]
plt.axhline(y=best_bo_value, color='b', linestyle='--', alpha=0.5)
plt.text(1, best_bo_value - 5, f'Best BO: {best_bo_value:.2f}', fontsize=11, color='blue')

# Mark the best point found by Random Search
best_rs_value = rs_running_best[-1]
plt.axhline(y=best_rs_value, color='orange', linestyle='--', alpha=0.5)
plt.text(70, best_rs_value + 5, f'Best RS: {best_rs_value:.2f}', fontsize=11, color='orange')

# Add vertical line showing when BO reaches near-optimal (5% of final)
near_optimal_threshold = best_bo_value * 1.05
for i, val in enumerate(bo_running_best):
    if val <= near_optimal_threshold:
        bo_convergence_iter = i
        break
plt.axvline(x=bo_convergence_iter, color='b', linestyle=':', alpha=0.5)
plt.text(bo_convergence_iter + 1, 200, f'BO convergence: iter {bo_convergence_iter}', fontsize=10, color='blue')

# Add vertical line showing when RS reaches near-optimal (5% of final)
for i, val in enumerate(rs_running_best):
    if val <= best_rs_value * 1.05:
        rs_convergence_iter = i
        break
plt.axvline(x=rs_convergence_iter, color='orange', linestyle=':', alpha=0.5)
plt.text(rs_convergence_iter + 1, 230, f'RS convergence: iter {rs_convergence_iter}', fontsize=10, color='orange')

# Customize the plot
plt.xlabel('Search Iteration', fontsize=14)
plt.ylabel('Performance (lower is better)', fontsize=14)
plt.title('Convergence Comparison for Apache System', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 350)

# Add improvement percentage annotation
improvement_pct = ((best_rs_value - best_bo_value) / best_rs_value) * 100
plt.annotate(f'Performance Improvement: {improvement_pct:.1f}%', 
             xy=(0.5, 0.03), xycoords='figure fraction', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5),
             fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('apache_convergence_comparison.png', dpi=300)
plt.show()