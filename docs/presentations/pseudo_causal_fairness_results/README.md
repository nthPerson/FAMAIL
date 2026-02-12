# Trajectory Modification Results Report (2/11/26)

This directory contains a comprehensive progress report on FAMAIL trajectory modification
results using the current pseudo-causal fairness formulation.

## Contents

- `index.html` - Interactive HTML report (open in browser)
- `generate_results.py` - Python script to run modification and generate results
- `results.json` - Generated results data (created by running the script)
- `README.md` - This file

## Generating Results

### Prerequisites

Ensure you have activated the FAMAIL conda environment and installed all dependencies:

```bash
conda activate famail
pip install -r trajectory_modification/requirements.txt
```

### Running the Script

From the project root directory:

```bash
python docs/presentations/pseudo_causal_fairness_results/generate_results.py
```

This script will:
1. Load trajectory data (up to 100 trajectories)
2. Fit the g(D) function using isotonic regression
3. Compute attribution scores (LIS + DCD) for all trajectories
4. Select top-10 trajectories by combined attribution score
5. Load the discriminator model from the specified checkpoint
6. Run the ST-iFGSM modification algorithm for each selected trajectory
7. Save all results to `results.json`

**Expected runtime:** 5-15 minutes depending on hardware (GPU recommended).

### Viewing the Report

After the script completes successfully:

1. Open `index.html` in a web browser
2. The report will automatically load and display results from `results.json`

Alternatively, you can use a local web server:

```bash
cd docs/presentations/pseudo_causal_fairness_results
python -m http.server 8000
# Then open http://localhost:8000 in your browser
```

## Report Contents

The HTML report includes:

### 1. Introduction
- Context on the pseudo-causal fairness formulation
- Note on ongoing reformulation work

### 2. Algorithm Overview
- Two-phase modification pipeline (Attribution + Modification)
- Objective function formulas for F_spatial, F_causal, F_fidelity
- ST-iFGSM algorithm pseudocode
- Soft cell assignment mechanism

### 3. Experiment Configuration
- Complete parameter settings used for this run
- Model checkpoints and selection criteria

### 4. Overall Results
- Metric cards showing before/after values with deltas
- Convergence statistics

### 5. Modification Details
- Table showing each trajectory's:
  - Original and modified pickup locations
  - Attribution scores (LIS, DCD, combined)
  - Convergence status and iteration count

### 6. Iteration Details
- Collapsible sections for each trajectory
- Full-precision iteration tables showing:
  - Objective L, F_spatial, F_causal, F_fidelity (10 decimal places)
  - Gradient norm and perturbation vectors

### 7. Visualizations
- Objective evolution line plots
- Perturbation magnitude histogram
- Convergence rate distribution

### 8. Causal Fairness Reformulation
- Explanation of why we're reformulating
- Option A1: Demographic Attribution
- Option B: Demographic Disparity
- Current status and decision criteria

## Configuration

The current configuration is:

```python
{
    'convergence_threshold': 1.0e-6,
    'epsilon': 3.0,
    'alpha': 0.10,
    'max_iterations': 50,
    'alpha_spatial': 0.33,
    'alpha_causal': 0.33,
    'alpha_fidelity': 0.34,
    'k': 10,
    'discriminator_checkpoint': 'discriminator/model/checkpoints/pass-seek_5000-20000_(84ident_72same_44diff)/best.pt',
    'gradient_mode': 'soft_cell',
    'temperature_annealing': True,
    'tau_max': 1.0,
    'tau_min': 0.1,
}
```

To modify parameters, edit `generate_results.py` and re-run.

## Color Scheme

The report uses the same color palette as the formulation validation presentation:

- Primary (Red): #A6192E
- Secondary (Gray): #CDCDC8
- Tertiary (Teal): #008080
- Charcoal: #2D2828

## Troubleshooting

### "File not found" error for discriminator checkpoint

Ensure the checkpoint path exists:

```bash
ls discriminator/model/checkpoints/pass-seek_5000-20000_*/best.pt
```

If missing, update the path in `generate_results.py`.

### "No module named 'trajectory_modification'" error

Make sure you're running from the project root directory and the module is on the Python path.

### Results not loading in browser

1. Check browser console for errors (F12)
2. Ensure `results.json` exists in the same directory as `index.html`
3. Try using a local web server (see "Viewing the Report" above)

## Notes

- The report is self-contained (except for CDN-loaded libraries: KaTeX, Plotly)
- Results are deterministic given fixed random seed in PyTorch/NumPy
- The g(D) function is fitted fresh each run using isotonic regression on loaded data
