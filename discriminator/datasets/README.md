# Discriminator Datasets

This directory contains generated trajectory pair datasets for training the discriminator model.

## Generating Datasets

Use the Dataset Generation Tool to create new datasets:

```bash
cd /home/robert/FAMAIL/discriminator/dataset_generation_tool
streamlit run app.py
```

After generating a full dataset, use the "Save to Directory" option to save it here with train/val splits.

## Expected Structure

```
datasets/
├── dataset_name_1/
│   ├── train.npz      # Training data
│   ├── val.npz        # Validation data
│   ├── test.npz       # Test data (optional)
│   └── metadata.json  # Configuration and statistics
├── dataset_name_2/
│   └── ...
└── README.md          # This file
```

## File Format

Each `.npz` file contains:
- `x1`: First trajectories `[N, L, 4]` (float32)
- `x2`: Second trajectories `[N, L, 4]` (float32)
- `label`: Binary labels `[N]` (float32) - 1=same agent, 0=different
- `mask1`: Validity mask for x1 `[N, L]` (bool)
- `mask2`: Validity mask for x2 `[N, L]` (bool)

## Training

Once you have a dataset, train with:

```bash
cd /home/robert/FAMAIL/discriminator/model
python train.py --data-dir ../datasets/your_dataset_name
```

Or use the Streamlit dashboard:

```bash
streamlit run training_dashboard.py
```
