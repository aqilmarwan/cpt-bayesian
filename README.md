# Configuration Performance Tuning with Bayesian Optimization

This project provides an improved method for configuration performance tuning that outperforms the random search baseline.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aqilmarwan/cpt-bayesian.git
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
cpt-bayesian/
├── bo_search_results/      # Bayesian optimization search results
├── bo_visualization_results/ # Visualizations of Bayesian optimization results
├── comparison_results/     # Comparison Analysis
   ├── comparison_convergence.py
   ├── comparison_improvement.py
   ├── comparison_methods.py
   ├── comparison_search.py
├── datasets/               # Contains input datasets (CSV files)
└── README.md               # This file
├── search_results/         # Original random search results
├── main.py                 # Original random search implementation
├── bayesian_optimization.py # Our improved Bayesian method
├── requirements.txt        # Python dependencies
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

## Search Results

| System          | Best Solution                           | Best Performance | Search Time (s) |
|---------------|---------------------------------|-----------------|----------------|
| LLVM.csv      | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 52285.4        | 50.60          |
| PostgreSQL.csv| [0, 1, 0, 0, 0, 128, 2, 256]   | 46022.8        | 38.91          |
| x264.csv      | [4, 1, 1, 1, 1, 0, 0, 0, 1, 0] | 21.556         | 40.81          |
| brotli.csv    | [14, 1]                         | 1.46           | 27.36          |
| storm.csv     | [2, 1, 4, 1000000, 1, 0, 1000, 10, 512, 100, 2, 29] | 2.7235e-05    | 47.98          |
| spear.csv     | [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1] | 0.0            | 47.62          |
| Apache.csv    | [0, 1, 2, 1, 0, 0, 512, 0]     | 30.7448        | 41.94          |
| 7z.csv        | [2, 1, 1, 0, 0, 80, 512, 4]    | 4305.8         | 34.76          |

## Comparison Summary

| Dataset     | Improvement | Random Best | Bayesian Best |
|------------|------------|-------------|--------------|
| storm      | nan%       | 0.00        | 0.00         |
| brotli     | 0.00%      | 1.46        | 1.46         |
| spear      | 100.00%    | 0.00        | 0.00         |
| PostgreSQL | 0.00%      | 45922.80    | 45922.80     |
| 7z         | 3.14%      | 4409.80     | 4271.20      |
| x264       | 0.65%      | 21.70       | 21.56        |
| LLVM       | 2.05%      | 53380.60    | 52285.40     |
| Apache     | 0.24%      | 30.82       | 30.74        |

===== Comparison Summary =====
Dataset         Improvement     Random Best     Bayesian Best  
------------------------------------------------------------
LLVM              8.17%       56936.20       52285.40
PostgreSQL       -0.04%       45922.80       45939.80
x264              3.91%          22.43          21.56
brotli            0.00%           1.46           1.46
storm             0.00%           0.00           0.00
spear           100.00%           0.00           0.00
Apache            1.65%          31.26          30.74
7z                2.48%        4323.00        4215.80
## Troubleshooting

If you encounter issues:

- Ensure all required packages are installed
- Check that your datasets have the correct format
- Verify that the system_name detection for minimization/maximization is correct