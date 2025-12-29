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

To verify actual coverage, click "Generate Full Dataset" and inspect the `agent_usage` in metadata.

## Example: Full Coverage Dataset

For training a discriminator on 50 agents with comprehensive coverage:

1. Check **"Per-agent counts mode"**
2. Set **# Positive Pairs per Agent** = 100 → 5,000 matching pairs
3. Set **# Negative Pairs per Agent Combination** = 10 → 24,500 non-matching pairs
4. Total dataset: **29,500 pairs** covering all agent combinations

This ensures the model sees:
- 100 examples of each agent matching with themselves
- 10 examples of each agent being compared to every other agent

## Notes

- Positive pairs are built from non-overlapping segments of the same agent.
- Negative pairs draw segments from different agents; in per-agent mode, every combination is covered.
- All generated sequences are aligned (padded or truncated) to a uniform length for export convenience.
