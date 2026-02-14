# Hyperparameter Search Guide

## Overview

The Demographic Explorer dashboard now includes a comprehensive hyperparameter search feature that helps you find the optimal model configuration for predicting service ratios based on demographics. This guide explains how to use the feature effectively.

## Features

### 1. GPU Acceleration

The dashboard automatically detects and utilizes GPU acceleration when available:

- **PyTorch Neural Networks**: Automatically uses CUDA for GPU training
- **XGBoost**: Leverages GPU for gradient boosting operations
- **Automatic Fallback**: Falls back to CPU if GPU is unavailable

#### Checking GPU Status

When you open the dashboard, the Model Training tab displays:
- ✅ GPU Available: [GPU Name] - when CUDA is detected
- ℹ️ GPU Not Available - using CPU - when running on CPU

### 2. Hyperparameter Search

Instead of manually tuning hyperparameters through trial and error, the search feature systematically explores the hyperparameter space to find optimal configurations.

#### Supported Models

- **Ridge Regression**: alpha (regularization strength)
- **Lasso Regression**: alpha
- **ElasticNet**: alpha, l1_ratio
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **Gradient Boosting**: n_estimators, max_depth, learning_rate, subsample
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Neural Networks** (sklearn & PyTorch): hidden_layer_sizes, learning_rate, alpha

## How to Use

### Step 1: Select Models

In the Model Training tab, check the models you want to train using the checkboxes.

### Step 2: Enable Hyperparameter Search

1. Scroll to the **Hyperparameter Search** section
2. Check the box "Enable Hyperparameter Search"
3. Choose your search method:
   - **Grid Search**: Exhaustively tries all combinations (thorough but slower)
   - **Random Search**: Samples N random combinations (faster, often sufficient)

### Step 3: Configure Search Space

For each selected model that supports hyperparameter search, click its expander to configure the search space:

#### Example: Ridge Regression

The default search space includes:
```
alpha: [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
```

You can customize by:
- Selecting a subset of values to try (faster search)
- Keeping all defaults for comprehensive search

#### Example: Neural Network

```
hidden_layer_sizes: [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)]
learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01]
alpha: [0.0001, 0.001, 0.01, 0.1]
```

### Step 4: Run the Search

1. Click "🔍 Run Hyperparameter Search"
2. Monitor progress:
   - Overall progress bar shows which model is being searched
   - Nested progress shows current hyperparameter combination being tested
3. Wait for completion

### Step 5: Review Results

After the search completes:

1. **Summary Section**: Shows best parameters and LODO R² for each model
2. **Top 5 Configurations**: Table showing the best-performing parameter combinations
3. **Comparison Table**: All models ranked by LODO R², with 🎯 indicating models that used hyperparameter search
4. **Model Details**: Expanders show the optimized parameters used

## Search Strategies

### Grid Search vs Random Search

**Grid Search:**
- Pros: Exhaustive, guaranteed to find best in search space
- Cons: Exponentially slower as search space grows
- Use when: Few hyperparameters, small ranges, need optimal result

**Random Search:**
- Pros: Much faster, often finds near-optimal solutions
- Cons: May miss global optimum
- Use when: Many hyperparameters, large ranges, time-constrained

### Recommended Search Sizes

For 10 districts (Leave-One-District-Out CV):

- **Quick exploration**: Random search with 10-20 samples
- **Standard**: Random search with 20-50 samples or small grid
- **Thorough**: Grid search with < 100 total combinations
- **Intensive**: Grid search with < 500 combinations (only if time permits)

## Performance Optimization

### GPU Acceleration Benefits

Expected speedups with GPU (vs CPU):

- **PyTorch Neural Network**: 2-5× faster
- **XGBoost**: 1.5-3× faster
- **Other models**: No GPU benefit (sklearn models are CPU-only)

### Reducing Search Time

1. **Use Random Search**: Start with 20 samples before trying grid search
2. **Reduce Search Space**:
   - For alpha: Try [0.01, 0.1, 1.0, 10.0] instead of 9 values
   - For n_estimators: Try [50, 100, 200] instead of [50, 100, 200, 300]
3. **Use GPU**: Enable GPU for PyTorch NN and XGBoost
4. **Reduce Data**: Use "July only" dataset for quick experiments, then validate on "All" dataset

## Interpreting Results

### LODO R² (Primary Metric)

- **> 0.30**: Strong - demographics explain substantial service variation
- **0.10 - 0.30**: Moderate - demographics have meaningful but limited effect
- **< 0.10**: Weak - demand-only model may be sufficient

### Overfit Gap

Best hyperparameters should minimize: `Train R² - LODO R²`

- **< 0.05**: Healthy generalization (green)
- **0.05 - 0.10**: Moderate concern (yellow)
- **> 0.10**: Severe overfitting (red)

### Best Practices

1. **Trust LODO R²**: It's the most reliable metric for model selection
2. **Check Per-District R²**: Ensure the model works across all districts, not just a few
3. **Compare to Baseline**: Check if g(D,x) outperforms g(D) alone
4. **Use Export**: Download the configuration JSON for reproducibility

## Example Workflow

### Scenario: Finding the best model for causal fairness

1. **Initial Exploration** (10 minutes):
   - Select: Ridge, Lasso, Random Forest, XGBoost
   - Enable hyperparameter search (Random, 20 samples)
   - Enable GPU
   - Run search

2. **Analyze Top Performers**:
   - Identify which model type performs best
   - Check overfit gap and per-district R²

3. **Refinement** (optional, 20 minutes):
   - Select only the best model type
   - Expand search space (Grid search or more random samples)
   - Re-run for fine-tuning

4. **Export Configuration**:
   - Go to Cross-Validation tab
   - Section 6: Export Model Configuration
   - Download JSON with best hyperparameters

5. **Implementation**:
   - Use exported configuration in `term.py`
   - Integrate into trajectory modification pipeline

## Troubleshooting

### "XGBoost not installed"

Install XGBoost with GPU support:

```bash
# CPU-only
pip install xgboost

# GPU (CUDA 11.x)
pip install xgboost --extra-index-url https://pypi.ngc.nvidia.com

# GPU (CUDA 12.x)
pip install xgboost
```

### "GPU available but not being used"

1. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Verify GPU checkbox is enabled in dashboard
3. Some models (Ridge, Lasso, Random Forest) don't support GPU

### Search takes too long

1. Reduce search space (use fewer hyperparameter values)
2. Switch to random search with fewer samples
3. Use smaller dataset (July only instead of All)
4. Enable GPU if available

### Low LODO R² for all models

This indicates demographics don't strongly predict service patterns. Consider:

1. Changing g(D) estimation method (try "isotonic" or "binning")
2. Using "all" period aggregation for more data per district
3. Adding more demographic features if available
4. This is actually informative - it means fairness is not strongly tied to demographics!

## Advanced Tips

### Custom Search Spaces

To define custom hyperparameter ranges, edit the expander configuration:

1. Enable hyperparameter search
2. Expand the model's configuration
3. Use multiselect to choose only the values you want to try

### Parallel Districts

The LODO cross-validation runs sequentially (10 folds for 10 districts). This is inherent to the Leave-One-District-Out strategy and cannot be parallelized within a single hyperparameter configuration.

However, hyperparameter combinations themselves run sequentially, not in parallel.

### Reproducibility

Random search uses `random.seed()` for reproducibility. To ensure exact reproduction:

1. Use the same search method (grid vs random)
2. Use the same number of random samples
3. Use the same search space configuration
4. Export and save the configuration JSON

## Integration with Causal Fairness Term

After finding the best hyperparameters:

1. **Export Configuration**: Download the JSON from Tab 4, Section 6
2. **Update term.py**: Modify `CausalFairnessTerm` to use optimized g(D,x) model
3. **Set Hyperparameters**: Use the `best_params` from the JSON in your model initialization
4. **Validate**: Run the fairness term computation and verify improved R² alignment

## References

- **LODO Cross-Validation**: Leave-One-District-Out is used to prevent data leakage across districts
- **GPU Acceleration**: Uses PyTorch CUDA and XGBoost GPU backends
- **Hyperparameter Optimization**: Based on scikit-learn's grid/random search patterns
