# FAMAIL: Fairness-Aware Multi-Agent Imitation Learning

<p align="center">
  <strong>Improving Spatial Fairness in Taxi Services Through Trajectory Editing</strong>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Research Goals](#research-goals)
- [Team](#team)
- [Methodology](#methodology)
- [Data Model](#data-model)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Related Publications](#related-publications)
- [License](#license)

---

## Overview

FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) is a research project at San Diego State University that addresses **spatial inequality in urban taxi services**. Using GPS trajectory data from Shenzhen, China, we develop trajectory editing techniques that modify expert driver trajectories to improve fairness metrics while maintaining trajectory authenticity.

### The Problem

Taxi services in cities like Shenzhen exhibit significant spatial inequality—certain areas receive disproportionately more or less service relative to demand. This inequality may correlate with socioeconomic factors such as neighborhood income levels, creating a systemic fairness issue in urban mobility.

### Our Approach

Rather than generating entirely new synthetic trajectories, FAMAIL focuses on **editing existing expert trajectories** to:

1. **Quantify unfairness** using spatial and causal fairness metrics
2. **Modify trajectories** to reduce unfairness while preserving realism
3. **Train fairer policies** using edited trajectories in an imitation learning framework

---

## Research Goals

1. **Develop a multi-objective function** that balances fairness with trajectory fidelity
2. **Create a trajectory modification algorithm** that iteratively improves fairness
3. **Build a discriminator model** to validate trajectory authenticity
4. **Train imitation learning agents** on fairness-edited trajectories

### Objective Function

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

| Term | Description |
|------|-------------|
| $F_{\text{causal}}$ | **Causal Fairness**: Measures demand-service alignment (service should match demand) |
| $F_{\text{spatial}}$ | **Spatial Fairness**: Measures equality of service distribution across regions (Gini-based) |
| $F_{\text{fidelity}}$ | **Trajectory Fidelity**: Ensures edited trajectories remain realistic (discriminator-based) |

---

## Team

| Role | Name | Affiliation |
|------|------|-------------|
| **Researcher** | Robert Ashe | San Diego State University |
| **Advisor** | Dr. Xin Zhang | San Diego State University |

---

## Methodology

### Trajectory Modification Algorithm

FAMAIL uses an iterative trajectory editing approach:

1. **Rank** trajectories by their contribution to global unfairness
2. **Identify** worst-offending trajectories using demand hierarchy filtering
3. **Modify** selected trajectories within constrained n×n grid neighborhoods
4. **Validate** fidelity using the discriminator (reject unrealistic edits)
5. **Recompute** fairness metrics after modifications
6. **Iterate** until convergence or improvement threshold is met

### Key Constraints

- **Local Boundedness**: Edits stay within ±n grid cells of original positions
- **Subtle Modifications**: Changes are small enough to maintain realism
- **Fidelity Gate**: Each edit must pass discriminator validation

---

## Data Model

### Study Area & Resolution

| Parameter | Value |
|-----------|-------|
| **Geographic Area** | Shenzhen, China |
| **Time Period** | July 2016 (with supplementary Aug-Sep 2016) |
| **Spatial Grid** | 48 × 90 cells (~0.01° × 0.01° per cell, ~1.1 km) |
| **Temporal Resolution** | 288 time buckets per day (5-minute intervals) |
| **Fleet Size** | 50 expert drivers (subset of 17,877 total) |
| **Days Analyzed** | Monday–Friday (Saturday and Sunday excluded) |

### Quantization Strategy

#### Spatial Quantization
```
Grid cell size: 0.01 degrees (~1.1 km)
Method: numpy.digitize with data-driven bounds
X dimension: 48 cells (longitude)
Y dimension: 90 cells (latitude)
```

#### Temporal Quantization
```
Time buckets: 5-minute intervals → 288 slots per day
Day indexing: Monday=1, Tuesday=2, ..., Friday=5
```

### Primary Datasets

| Dataset | Description | Key Fields |
|---------|-------------|------------|
| `all_trajs.pkl` | Feature-augmented trajectories (50 drivers) | 126-element state vectors with location, time, POI distances, traffic features |
| `pickup_dropoff_counts.pkl` | Aggregated pickup/dropoff events | Counts by `(x_grid, y_grid, time_bucket, day)` |
| `latest_traffic.pkl` | Traffic conditions | Speed, volume, wait times per cell/time |
| `latest_volume_pickups.pkl` | Pickup volumes with neighborhoods | Extended pickup data with surrounding cell context |

### State Vector Schema

Each trajectory state is a 126-element vector:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `x_grid` | Grid x-coordinate (longitude) |
| 1 | `y_grid` | Grid y-coordinate (latitude) |
| 2 | `time_bucket` | Time of day [0-287] |
| 3 | `day_index` | Day of week |
| 4-24 | POI distances | Manhattan distances to 21 points of interest |
| 25-49 | Pickup counts | Normalized pickups in 5×5 window |
| 50-74 | Traffic volume | Normalized traffic volumes in 5×5 window |
| 75-99 | Traffic speed | Normalized speeds in 5×5 window |
| 100-124 | Traffic wait | Normalized wait times in 5×5 window |
| 125 | `action_code` | Movement action label (0-9) |

---

## Repository Structure

```
FAMAIL/
├── README.md                    # This file
├── DATASETS.md                  # Dataset generation instructions
│
├── objective_function/          # 🎯 FAMAIL objective function implementation
│   ├── spatial_fairness/        # Gini-based spatial fairness term
│   ├── causal_fairness/         # R²-based causal fairness term
│   ├── fidelity/                # Discriminator-based fidelity term
│   └── docs/                    # Comprehensive specifications
│
├── discriminator/               # 🔍 Trajectory authenticity discriminator
│   ├── model/                   # ST-SiameseNet implementation
│   └── dataset_generation_tool/ # Training data generation
│
├── pickup_dropoff_counts/       # 📍 Pickup/dropoff event processor
│   ├── processor.py             # Core processing logic
│   └── app.py                   # Streamlit dashboard
│
├── active_taxis/                # 🚕 Active taxi count generator
│   ├── active_taxis.py          # Core computation
│   └── app.py                   # Streamlit dashboard
│
├── data_dictionary/             # 📖 Dataset documentation
│   ├── dictionaries/            # Field-level documentation per dataset
│   └── explorers/               # Interactive data exploration notebooks
│
├── data/                        # 📊 Processed data and samples
│   ├── dataset_samples/         # Small samples for testing
│   ├── Processed_Data/          # Data processing notebooks
│   └── visualization/           # Visualization outputs
│
├── cGAIL_data_and_processing/   # 📓 cGAIL data exploration notebooks
│
├── source_data/                 # 📁 Source datasets (gitignored)
│
└── raw_data/                    # 📁 Raw GPS data (gitignored)
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| **objective_function/** | Core FAMAIL objective function with spatial fairness, causal fairness, and fidelity terms. Includes development plans, specifications, and implementation code. |
| **discriminator/** | ST-SiameseNet-based discriminator that validates trajectory authenticity. Used as the fidelity measure in the objective function. |
| **pickup_dropoff_counts/** | Data processing tool that counts pickup/dropoff events from raw GPS data, aggregated by spatiotemporal keys. |
| **active_taxis/** | Generates datasets counting active taxis in n×n neighborhoods for each cell/time period. Required for computing service rates. |
| **data_dictionary/** | Comprehensive documentation of all datasets including field descriptions, data types, and value ranges. |
| **data/** | Processed datasets, samples, and visualization outputs. |
| **cGAIL_data_and_processing/** | Jupyter notebooks for exploring cGAIL trajectory data and features. |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Conda (recommended for environment management)

### Installation

```bash
# Clone the repository
git clone https://github.com/nthPerson/FAMAIL.git
cd FAMAIL

# Create conda environment
conda create -n famail python=3.10
conda activate famail

# Install dependencies (per component)
pip install -r objective_function/requirements.txt
pip install -r discriminator/model/requirements.txt
pip install -r pickup_dropoff_counts/requirements.txt
```

### Quick Start

1. **Explore the data**: Start with the data dictionary and explorer notebooks
   ```bash
   jupyter notebook data_dictionary/explorers/all_trajs_explorer.ipynb
   ```

2. **Process pickup/dropoff counts**: Generate aggregated event data
   ```bash
   cd pickup_dropoff_counts
   streamlit run app.py
   ```

3. **Train the discriminator**: Build the trajectory authenticity model
   ```bash
   cd discriminator/model
   python train.py --data-dir ../datasets/traj_pair_5000pos_5000neg_80-20_split
   ```

---

## Related Publications

### Foundation Papers

1. **cGAIL Framework**: Zhang, X., et al. "Conditional Generative Adversarial Imitation Learning for Taxi Driver Trajectories"
   - Data source and baseline imitation learning model

2. **ST-iFGSM Trajectory Modification**: Zhang, X., et al. "ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM"
   - Foundation for trajectory modification approach

3. **Spatial Fairness**: Su, L., Yan, Z., & Cao, J. (2018). "Uncovering Spatial Inequality in Taxi Services in the Context of a Subsidy War among E-Hailing Apps"
   - Gini coefficient approach for measuring spatial inequality

4. **Causal Fairness**: Kilbertus, N., et al. "Avoiding Discrimination through Causal Reasoning"
   - Causal framework for fairness analysis

---

## License

This project is part of academic research at San Diego State University.

---

<p align="center">
  <em>The Fairness-Aware Multi Agent Imitation Learning (FAMAIL) Project -- 2025-2026</em>
</p>
