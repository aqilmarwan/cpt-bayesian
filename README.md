# Configuration Performance Tuning with Bayesian Optimization

This project provides an improved method for configuration performance tuning that outperforms the random search baseline.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/config-tuning.git
   cd config-tuning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies are:
   - numpy
   - pandas
   - matplotlib
   - scikit-learn

## Directory Structure

Ensure your project has the following structure:
```
project-folder/
├── datasets/               # Contains input datasets (CSV files)
├── search_results/         # Original random search results
├── bo_search_results/      # Bayesian optimization search results
├── bo_visualization_results/ # Visualizations of Bayesian optimization results
├── comparison_results/     # Comparison between methods
├── main.py                 # Original random search implementation
├── bayesian_optimization.py # Our improved Bayesian method
├── compare_methods.py      # Framework for comparing methods
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Running the Code

### 1. Run the Bayesian Optimization Method

```bash
python bayesian_optimization.py
```

This will:
- Process all datasets in the `datasets` folder
- Apply Bayesian optimization to find optimal configurations
- Save results to `bo_search_results` folder
- Generate visualizations in `bo_visualization_results` folder

### 2. Compare with Random Search Baseline

```bash
python compare_methods.py --budget 100 --repetitions 5
```

This will:
- Run both methods multiple times on each dataset
- Generate comparative visualizations
- Produce a summary report of improvements

### 3. Examining the Results

- Check the generated CSV files in `bo_search_results` for detailed results
- View the visualizations in `bo_visualization_results` to see performance over iterations
- Examine `comparison_results/comparison_summary.csv` for a quantitative comparison
- Look at the bar charts in `comparison_results/` for visual comparison

## Configuration Parameters

You can adjust the following parameters in the code or via command-line arguments:

- `--budget`: Number of configurations to evaluate (default: 100)
- `--repetitions`: Number of repetitions for statistical significance (default: 5)
- `--output_dir`: Directory to save comparison results

## Interpreting the Results

- For minimization problems: Lower values are better
- For maximization problems: Higher values are better
- The improvement percentage shows how much Bayesian optimization outperforms random search
- The running best performance curve shows the convergence behavior

## Customizing the Method

You can customize the Bayesian optimization approach by adjusting:

1. The exploration-exploitation balance (modify the % of neighborhood vs random sampling)
2. The GP model parameters in `GaussianProcessRegressor`
3. The acquisition function (currently using Thompson Sampling)

## Troubleshooting

If you encounter issues:

- Ensure all required packages are installed
- Check that your datasets have the correct format
- Verify that the system_name detection for minimization/maximization is correct