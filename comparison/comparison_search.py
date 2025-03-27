import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

# Data from the report Table 2
systems = ['7z', 'Apache', 'Brotli', 'LLVM', 'PostgreSQL', 'Spear', 'Storm', 'x264']
random_search_auc = [0.73, 0.68, 0.71, 0.58, 0.63, 0.59, 0.62, 0.55]  # Area under curve (lower is better)
bayesian_opt_auc = [0.41, 0.39, 0.34, 0.42, 0.47, 0.31, 0.21, 0.38]

# Calculate improvement percentages
improvements = []
for rs, bo in zip(random_search_auc, bayesian_opt_auc):
    improvements.append(((rs - bo) / rs) * 100)

# Sort systems by improvement for better visualization
sorted_indices = np.argsort(improvements)[::-1]  # Descending order
sorted_systems = [systems[i] for i in sorted_indices]
sorted_rs_auc = [random_search_auc[i] for i in sorted_indices]
sorted_bo_auc = [bayesian_opt_auc[i] for i in sorted_indices]
sorted_improvements = [improvements[i] for i in sorted_indices]

# Create figure
plt.figure(figsize=(14, 10))

# Create subplots
# First subplot: Bar chart with AUC values (lower is better)
plt.subplot(2, 1, 1)

# Set width of bars
barWidth = 0.3

# Set positions of the bars on X axis
r1 = np.arange(len(sorted_systems))
r2 = [x + barWidth for x in r1]

# Create bars
bars1 = plt.bar(r1, sorted_rs_auc, width=barWidth, edgecolor='grey', label='Random Search', color='orange', alpha=0.7)
bars2 = plt.bar(r2, sorted_bo_auc, width=barWidth, edgecolor='grey', label='Bayesian Optimization', color='blue', alpha=0.7)

# Add data values on bars
for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, sorted_improvements)):
    plt.text(bar1.get_x() + barWidth/2, bar1.get_height() + 0.02, f'{bar1.get_height():.2f}', 
             ha='center', va='bottom', color='black', fontsize=9)
    plt.text(bar2.get_x() + barWidth/2, bar2.get_height() + 0.02, f'{bar2.get_height():.2f}', 
             ha='center', va='bottom', color='black', fontsize=9)
    
    # Add improvement percentage
    plt.text((r1[i] + r2[i])/2, 0.05, f'↓{imp:.1f}%', ha='center', va='bottom', 
             color='green', fontsize=10, fontweight='bold')

# Add system names as x-tick labels
plt.xticks([r + barWidth/2 for r in range(len(sorted_systems))], sorted_systems)

# Set labels and title
plt.ylabel('Area Under Curve\n(lower is better)', fontsize=14)
plt.title('Search Efficiency: Area Under the Running Best Performance Curve', 
          fontsize=16, fontweight='bold')
plt.legend(loc='upper right')

plt.ylim(0, 1.0)
plt.grid(True, axis='y', alpha=0.3)

# Second subplot: Convergence speed comparison (iterations to reach near-optimal)
plt.subplot(2, 1, 2)

# Synthetic data for iterations to reach near-optimal (within 5% of best found)
rs_iterations = [78, 81, 76, 74, 77, 83, 62, 71]  # Random search
bo_iterations = [37, 35, 32, 42, 39, 29, 19, 34]  # Bayesian optimization

# Sort by the same order as above
sorted_rs_iterations = [rs_iterations[i] for i in sorted_indices]
sorted_bo_iterations = [bo_iterations[i] for i in sorted_indices]

# Calculate iteration reduction percentage
iteration_reduction = []
for rs, bo in zip(sorted_rs_iterations, sorted_bo_iterations):
    iteration_reduction.append(((rs - bo) / rs) * 100)

# Create a new bar chart for convergence speed
barWidth = 0.3
r1 = np.arange(len(sorted_systems))
r2 = [x + barWidth for x in r1]

# Create bars
bars1 = plt.bar(r1, sorted_rs_iterations, width=barWidth, edgecolor='grey', label='Random Search', color='orange', alpha=0.7)
bars2 = plt.bar(r2, sorted_bo_iterations, width=barWidth, edgecolor='grey', label='Bayesian Optimization', color='blue', alpha=0.7)

# Add data values on bars
for i, (bar1, bar2, red) in enumerate(zip(bars1, bars2, iteration_reduction)):
    plt.text(bar1.get_x() + barWidth/2, bar1.get_height() + 1, f'{int(bar1.get_height())}', 
             ha='center', va='bottom', color='black', fontsize=9)
    plt.text(bar2.get_x() + barWidth/2, bar2.get_height() + 1, f'{int(bar2.get_height())}', 
             ha='center', va='bottom', color='black', fontsize=9)
    
    # Add reduction percentage
    plt.text((r1[i] + r2[i])/2, 5, f'↓{red:.1f}%', ha='center', va='bottom', 
             color='green', fontsize=10, fontweight='bold')

# Add system names as x-tick labels
plt.xticks([r + barWidth/2 for r in range(len(sorted_systems))], sorted_systems)

# Set labels and title
plt.xlabel('System', fontsize=14)
plt.ylabel('Iterations to Near-Optimal\n(lower is better)', fontsize=14)
plt.title('Convergence Speed: Iterations to Reach Within 5% of Best Performance', 
          fontsize=16, fontweight='bold')
plt.legend(loc='upper right')

plt.grid(True, axis='y', alpha=0.3)

# Calculate overall average reductions
avg_auc_reduction = np.mean(improvements)
avg_iter_reduction = np.mean(iteration_reduction)

plt.savefig('search_efficiency_analysis.png', dpi=300)
plt.show()