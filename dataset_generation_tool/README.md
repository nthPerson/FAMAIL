# Trajectory Pair Dataset Generator

Streamlit-based tool for producing labeled trajectory pairs for the ST-SiameseNet discriminator. Implements the requirements from the "Trajectory Pair Dataset Generator" design doc.

## Quick start
```bash
pip install -r discriminator/dataset_generation_tool/requirements.txt
streamlit run discriminator/dataset_generation_tool/app.py
```

## Sampling Modes

The tool supports **two distinct sampling modes** controlled by the **"Per-agent counts mode"** checkbox:

### Default Mode: Total Pair Counts

When **unchecked** (default):
- **# Positive Pairs (Total)**: The absolute number of matching (same-agent) pairs in the dataset.
- **# Negative Pairs (Total)**: The absolute number of non-matching (different-agent) pairs in the dataset.

Example: Setting 500 positive and 500 negative produces exactly 1,000 total pairs.

> **Note:** In this mode, `Ensure every agent appears` checkbox helps guarantee each agent is represented at least once, but does not guarantee equal representation or complete cross-agent coverage.

### Per-Agent Counts Mode (for Comprehensive Discriminator Training)

When **checked**:
- **# Positive Pairs per Agent**: Generates this many matching pairs **for each agent**.
  - Total positive pairs = `N × num_agents`
  - Example: 100 positive pairs × 50 agents = **5,000 total matching pairs**
  
- **# Negative Pairs per Agent Combination**: Generates this many pairs **for each (agent_i, agent_j) combination** where `i ≠ j`.
  - Total negative pairs = `N × num_agents × (num_agents - 1)`
  - Example: 10 negative pairs × 50 × 49 = **24,500 total non-matching pairs**

This mode ensures:
✅ **All 50 agents are represented** in both positive and negative samples  
✅ **Every agent-pair combination is covered** in negative samples  
✅ **Balanced training data** for the discriminator to learn all agent identities  

**Use this mode when training a discriminator that must distinguish between all agents.**

## Configuration Options

The sidebar lets you set:
- **Per-agent counts mode**: Toggle between total counts and per-agent counts
- **Positive/negative pair counts** and random seed
- **Agent sampling distribution** (proportional or uniform) — only applies in total mode
- **Positive strategy** (random vs sequential non-overlapping segments)
- **Negative strategy** (random vs round-robin agents) — only applies in total mode
- **Concatenated trajectory span** (1–7 days)
- **Optional extra feature slice** beyond indices 0–3 (defaults to none)
- **Padding/truncation mode** (defaults to truncate to shorter; pad or fixed length also available)

## Data Assumptions

- Input pickle (`all_trajs.pkl`) is a `dict` of 50 agents → list of trajectories.
- Each trajectory is `(T, 126)`; indices 0–3 are `x_grid, y_grid, time_bucket, day_index`.
- Trajectories per agent are assumed to be in chronological order.

## Output Format

### Array Outputs

The generated dataset contains the following NumPy arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `x1` | `[N, L, F]` | First trajectory in each pair |
| `x2` | `[N, L, F]` | Second trajectory in each pair |
| `label` | `[N]` | Pair labels: **0** = same agent (positive), **1** = different agents (negative) |
| `mask1` | `[N, L]` | Validity mask for x1: **1** = real data, **0** = padding |
| `mask2` | `[N, L]` | Validity mask for x2 |

Where:
- **N** = total number of pairs
- **L** = sequence length (padded/truncated to uniform length)
- **F** = number of features (minimum 4: indices 0-3)

### Feature Indices

The first 4 features (always included) are:

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 0 | `x_grid` | Grid x-coordinate | 0-49 |
| 1 | `y_grid` | Grid y-coordinate | 0-89 |
| 2 | `time_bucket` | Time of day | 1-288 (5-min buckets) |
| 3 | `day_index` | Day of week | 1-6 |

Additional features (indices 4-125) can be optionally included via the feature slice settings.

### Metadata JSON

The metadata includes:

```json
{
  "config": { ... },           // Generation configuration used
  "counts": {
    "total_pairs": 22,         // Total pairs in dataset
    "positive_pairs": 12,      // Same-agent pairs (label=0)
    "negative_pairs": 10       // Different-agent pairs (label=1)
  },
  "length_stats": {
    "x1": { "min": 62, "max": 760, "mean": 414.09, "p50": 466, "p90": 574.8, "p95": 593.05 },
    "x2": { "min": 3, "max": 582, ... },
    "combined": { ... },
    "padded_length": 541       // Final uniform sequence length
  },
  "agent_usage": {
    "0": { "pos": 2, "neg": 1 },  // Agent 0 appears in 2 positive, 1 negative pairs
    "1": { "pos": 2, "neg": 1 },
    ...
  },
  "dataset_hash": "99ddd9a0..."  // Hash for dataset identification
}
```

### Download Formats

- **`.npz`** — NumPy compressed archive (always available)
- **`.pt`** — PyTorch tensors (if PyTorch installed)
- **`.json`** — Small sample for inspection

## Preview vs Full Generation

⚠️ **Important**: Preview mode uses **capped total counts** for fast iteration, even when "Per-agent counts mode" is enabled.

| Mode | Positive Pairs | Negative Pairs | Per-Agent Coverage |
|------|----------------|----------------|-------------------|
| **Preview** | min(configured, preview_cap) | min(configured, preview_cap) | ❌ Not guaranteed |
| **Full Dataset** | As configured (total or per-agent) | As configured (total or per-agent) | ✅ Guaranteed in per-agent mode |

To verify actual coverage, click "Generate Full Dataset" and use the **Dataset Validation Report**.

## Dataset Validation Feature

After generating a full dataset, the tool automatically displays a comprehensive **Dataset Validation Report** that includes:

### Validation Checks

The validation report runs automatic checks on the generated dataset:

| Check | Description |
|-------|-------------|
| Positive/Negative pair counts | Verifies metadata counts match actual label distribution |
| Array shape consistency | Ensures x1, x2, mask1, mask2, label arrays have compatible shapes |
| Feature count | Validates the number of features matches configuration |
| Sequence length | Confirms all sequences are padded to the expected length |
| Label validity | Ensures all labels are 0 (positive) or 1 (negative) |
| Agent coverage | (Per-agent mode) Checks that all agents are represented |
| Min pairs per agent | (Per-agent mode) Verifies minimum positive pairs per agent |

### Agent Distribution Histograms

Interactive visualizations showing:

1. **Positive Distribution** — Bar chart of positive (same-agent) pairs per agent
2. **Negative Distribution** — Bar chart of negative (different-agent) pairs per agent  
3. **Total Distribution** — Combined view of total appearances per agent
4. **Stacked Comparison** — Positive vs negative breakdown per agent

Each histogram includes statistics:
- Min, Max, Mean, Standard Deviation

### Coverage Analysis

Summary metrics for agent coverage:
- Number of agents with positive pairs
- Number of agents with negative pairs
- Number of agents with both types
- List of under-represented agents (if any)

### Sequence Length Analysis

Histograms of actual trajectory lengths (from mask sums):
- x1 length distribution by label type
- x2 length distribution by label type

### Configuration Summary

Full display of all generation settings used:
- Sampling mode (total vs per-agent)
- Pair counts and strategies
- Feature slice configuration
- Padding/truncation settings
- Random seed used

## Example: Full Coverage Dataset

For training a discriminator on 50 agents with comprehensive coverage:

1. Check **"Per-agent counts mode"**
2. Set **# Positive Pairs per Agent** = 100 → 5,000 matching pairs
3. Set **# Negative Pairs per Agent Combination** = 10 → 24,500 non-matching pairs
4. Total dataset: **29,500 pairs** covering all agent combinations
5. Click **"Generate Full Dataset"** and review the validation report

This ensures the model sees:
- 100 examples of each agent matching with themselves
- 10 examples of each agent being compared to every other agent

The validation report will confirm:
✅ All 50 agents represented  
✅ Each agent has ≥100 positive pairs  
✅ Each agent participates in ~980 negative pairs (10 × 49 as anchor + 10 × 49 as other)

## Notes

- Positive pairs are built from non-overlapping segments of the same agent.
- Negative pairs draw segments from different agents; in per-agent mode, every combination is covered.
- All generated sequences are aligned (padded or truncated) to a uniform length for export convenience.
