# all_demographics_by_district.csv Data Dictionary

## Overview

The `all_demographics_by_district.csv` dataset contains demographic, economic, and housing data for 10 districts in Shenzhen, China. It provides district-level population breakdowns, economic indicators, and housing price information used as contextual features in the FAMAIL project. This dataset was processed and renamed from the raw `demographics_by_district.csv` source file to provide cleaner, more programmatically accessible column names.

**File Format:** CSV (comma-separated values)  
**Data Type:** Tabular (rows × columns)  
**Number of Records:** 10 districts  
**Number of Fields:** 14  
**Source Data:** `raw_data/demographics_by_district.csv`  
**Processing Notebook:** `data/demographic_data/create_demographic_metadata.ipynb`

---

## Data Structure

### Loading the Data

```python
import pandas as pd

# Load the dataset
demographics_df = pd.read_csv('path/to/all_demographics_by_district.csv', index_col=0)
print(f'Number of districts: {len(demographics_df)}')  # Output: 10
print(f'Number of columns: {len(demographics_df.columns)}')  # Output: 14

# Preview the data
demographics_df.head()
```

---

## Column Renaming Reference

The raw source file (`demographics_by_district.csv`) uses descriptive English column headers. These were renamed to shorter, code-friendly names during processing. The mapping is as follows:

| Original Column Name | Renamed Column Name | Notes |
|---|---|---|
| `District` | `DistrictName` | Identifier field |
| `Land Area (sq km)` | `AreaKm2` | Units preserved |
| `Year-end Permanent Population (10,000 persons)` | `YearEndPermanentPop10k` | Units: 10,000 persons |
| `Permanent Registered Population (10,000)` | `RegisteredPermanentPop10k` | Units: 10,000 persons |
| `Permanent Non-registered Population (10,000)` | `NonRegisteredPermanentPop10k` | Units: 10,000 persons |
| `Population Density (persons per sq km)` | `PopDensityPerKm2` | Units: persons/km² |
| `Household Registration Population (10,000 persons)` | `HouseholdRegisteredPop10k` | Units: 10,000 persons |
| `Men (10,000 persons)` | `MalePop10k` | Units: 10,000 persons |
| `Women (10,000 persons)` | `FemalePop10k` | Units: 10,000 persons |
| `Sex Ratio (equal men and women = 100)` | `SexRatio100` | Baseline: 100 = equal |
| `Employee Compensation Payable (100 million yuan)` | `EmployeeCompensation100MYuan` | Units: 100 million yuan |
| `Average Number of Employed Persons (person)` | `AvgEmployedPersons` | Units: persons |
| `Average Housing Price (per sq/m)` | `AvgHousingPricePerSqM` | Units: yuan/m² |
| `GDP (10000 yuan)` | `GDPin10000Yuan` | Units: 10,000 yuan |

---

## Column Schema

| Column | Data Type | Unique Values | Missing Values | Numeric | Description |
|--------|-----------|---------------|----------------|---------|-------------|
| `DistrictName` | `object` | 10 | 0 | No | District name (categorical identifier) |
| `AreaKm2` | `float64` | 10 | 0 | Yes | Land area in square kilometers |
| `YearEndPermanentPop10k` | `float64` | 10 | 0 | Yes | Year-end permanent population (in 10,000 persons) |
| `RegisteredPermanentPop10k` | `float64` | 10 | 0 | Yes | Permanent registered (hukou) population (in 10,000 persons) |
| `NonRegisteredPermanentPop10k` | `float64` | 10 | 0 | Yes | Permanent non-registered population (in 10,000 persons) |
| `PopDensityPerKm2` | `int64` | 10 | 0 | Yes | Population density (persons per km²) |
| `HouseholdRegisteredPop10k` | `float64` | 10 | 0 | Yes | Household registration population (in 10,000 persons) |
| `MalePop10k` | `float64` | 10 | 0 | Yes | Male population (in 10,000 persons) |
| `FemalePop10k` | `float64` | 10 | 0 | Yes | Female population (in 10,000 persons) |
| `SexRatio100` | `float64` | 10 | 0 | Yes | Sex ratio (males per 100 females; 100 = equal) |
| `EmployeeCompensation100MYuan` | `float64` | 10 | 0 | Yes | Total employee compensation payable (in 100 million yuan) |
| `AvgEmployedPersons` | `int64` | 10 | 0 | Yes | Average number of employed persons |
| `AvgHousingPricePerSqM` | `float64` | 6 | 0 | Yes | Average housing price per square meter (yuan/m²) |
| `GDPin10000Yuan` | `int64` | 10 | 0 | Yes | Gross Domestic Product (in 10,000 yuan) |

---

## Detailed Field Descriptions

### `DistrictName`
- **Type:** String (object)
- **Description:** Name of the Shenzhen district
- **Values:** Futian, Luohu, Yantian, Nanshan, Bao'an, Longgang, Longhua, Pingshan, Guangming, Dapeng
- **Role:** Primary identifier / index field

### `AreaKm2`
- **Type:** Float
- **Description:** Total land area of the district in square kilometers
- **Range:** 74.91 (Yantian) – 396.61 (Bao'an)
- **Unit:** km²

### `YearEndPermanentPop10k`
- **Type:** Float
- **Description:** Total year-end permanent population, including both registered and non-registered residents
- **Range:** 14.09 (Dapeng) – 301.71 (Bao'an)
- **Unit:** 10,000 persons
- **Note:** `YearEndPermanentPop10k ≈ RegisteredPermanentPop10k + NonRegisteredPermanentPop10k`

### `RegisteredPermanentPop10k`
- **Type:** Float
- **Description:** Permanent population with local household registration (hukou)
- **Range:** 3.93 (Dapeng) – 95.35 (Futian)
- **Unit:** 10,000 persons

### `NonRegisteredPermanentPop10k`
- **Type:** Float
- **Description:** Permanent residents without local household registration (migrant workers, temporary residents)
- **Range:** 10.16 (Dapeng) – 253.96 (Bao'an)
- **Unit:** 10,000 persons
- **Note:** Bao'an and Longgang have significantly higher non-registered populations, reflecting large migrant worker communities

### `PopDensityPerKm2`
- **Type:** Integer
- **Description:** Population density calculated as persons per square kilometer
- **Range:** 477 (Dapeng) – 19,091 (Futian)
- **Unit:** persons/km²
- **Note:** Wide variation reflects urban core vs. peripheral districts

### `HouseholdRegisteredPop10k`
- **Type:** Float
- **Description:** Population registered under the household registration system
- **Range:** 3.97 (Dapeng) – 98.97 (Futian)
- **Unit:** 10,000 persons
- **Note:** Closely tracks `RegisteredPermanentPop10k` but may differ slightly due to registration timing

### `MalePop10k`
- **Type:** Float
- **Description:** Male population of the district
- **Range:** 2.13 (Dapeng) – 49.40 (Futian)
- **Unit:** 10,000 persons

### `FemalePop10k`
- **Type:** Float
- **Description:** Female population of the district
- **Range:** 1.84 (Dapeng) – 49.57 (Futian)
- **Unit:** 10,000 persons

### `SexRatio100`
- **Type:** Float
- **Description:** Ratio of males to females, where 100 indicates equal numbers. Values above 100 indicate more males than females.
- **Range:** 99.61 (Luohu) – 115.02 (Dapeng)
- **Unit:** Males per 100 females
- **Note:** Industrial districts (Nanshan, Dapeng) tend to have higher ratios, likely due to male-dominated industries

### `EmployeeCompensation100MYuan`
- **Type:** Float
- **Description:** Total employee compensation payable in the district
- **Range:** 2.58 (Pingshan) – 552.17 (Futian)
- **Unit:** 100 million yuan (i.e., 1 = 100,000,000 yuan)

### `AvgEmployedPersons`
- **Type:** Integer
- **Description:** Average number of employed persons in the district
- **Range:** 4,526 (Pingshan) – 557,197 (Futian)
- **Unit:** Persons

### `AvgHousingPricePerSqM`
- **Type:** Float
- **Description:** Average housing price per square meter in the district
- **Unique Values:** 6 (some districts share the same average)
- **Range:** 48,108.0 (Luohu) – 51,138.67 (Longgang, Pingshan, Dapeng)
- **Unit:** Yuan per m²
- **Note:** Only 6 unique values across 10 districts, suggesting grouping or averaging across sub-regions

### `GDPin10000Yuan`
- **Type:** Integer
- **Description:** Gross Domestic Product of the district
- **Range:** 3,074,578 (Dapeng) – 38,452,711 (Nanshan)
- **Unit:** 10,000 yuan (i.e., 1 = 10,000 yuan)

---

## Districts

The dataset covers 10 districts of Shenzhen, a major city in Guangdong Province, China:

| District | Area (km²) | Population (10k) | Pop. Density (/km²) | GDP (10k yuan) |
|----------|-----------|-------------------|---------------------|----------------|
| Futian | 78.66 | 150.17 | 19,091 | 35,572,870 |
| Luohu | 78.75 | 100.40 | 12,749 | 19,724,939 |
| Yantian | 74.91 | 22.65 | 3,024 | 5,375,327 |
| Nanshan | 187.47 | 135.63 | 7,235 | 38,452,711 |
| Bao'an | 396.61 | 301.71 | 7,607 | 30,038,215 |
| Longgang | 388.22 | 214.38 | 5,522 | 31,790,883 |
| Longhua | 175.58 | 154.94 | 8,824 | 18,569,841 |
| Pingshan | 166.31 | 40.79 | 2,453 | 5,060,882 |
| Guangming | 155.44 | 56.08 | 3,608 | 7,265,766 |
| Dapeng | 295.32 | 14.09 | 477 | 3,074,578 |

---

## Usage Examples

### Basic Analysis

```python
import pandas as pd

df = pd.read_csv('all_demographics_by_district.csv', index_col=0)

# Find the most densely populated district
densest = df.loc[df['PopDensityPerKm2'].idxmax()]
print(f"Most dense district: {densest['DistrictName']} ({densest['PopDensityPerKm2']:,} per km²)")

# Calculate total population across all districts
total_pop = df['YearEndPermanentPop10k'].sum()
print(f"Total Shenzhen population: {total_pop * 10000:,.0f}")
```

### Registered vs. Non-Registered Population Ratio

```python
# Analyze the proportion of non-registered (migrant) population
df['NonRegisteredRatio'] = df['NonRegisteredPermanentPop10k'] / df['YearEndPermanentPop10k']
print(df[['DistrictName', 'NonRegisteredRatio']].sort_values('NonRegisteredRatio', ascending=False))

# Bao'an and Longgang will have the highest non-registered ratios (~84% and ~75%)
```

### Economic Indicators

```python
# GDP per capita (approximate)
df['GDPperCapita'] = (df['GDPin10000Yuan'] * 10000) / (df['YearEndPermanentPop10k'] * 10000)
print(df[['DistrictName', 'GDPperCapita']].sort_values('GDPperCapita', ascending=False))
```

---

## Data Quality Notes

### Completeness

- **Missing Values:** None — all 14 columns have complete data across all 10 districts
- **Unique Values:** All columns have 10 unique values except `AvgHousingPricePerSqM` (6 unique values), indicating some districts share averaged housing price figures

### Known Limitations

1. **Temporal Snapshot:** Single point-in-time data; no temporal dimension
2. **Granularity:** District-level only; no sub-district or neighborhood-level breakdowns
3. **Housing Price Grouping:** Only 6 distinct housing price values for 10 districts suggests averaging or grouping
4. **Unit Conventions:** Mixed units (10,000 persons, 100 million yuan, 10,000 yuan) require careful handling when computing derived metrics

### Data Validation

```python
# Validate population consistency
for _, row in df.iterrows():
    registered = row['RegisteredPermanentPop10k']
    non_registered = row['NonRegisteredPermanentPop10k']
    total = row['YearEndPermanentPop10k']
    assert abs((registered + non_registered) - total) < 0.1, \
        f"Population mismatch for {row['DistrictName']}"

# Validate sex ratio consistency
for _, row in df.iterrows():
    computed_ratio = (row['MalePop10k'] / row['FemalePop10k']) * 100
    assert abs(computed_ratio - row['SexRatio100']) < 1.0, \
        f"Sex ratio mismatch for {row['DistrictName']}"
```

---

## Relationship to Other Datasets

### Connection to Spatial Grid

The district-level demographics provide contextual features for the FAMAIL spatial grid. Districts map to regions of the 48×90 grid used in trajectory and traffic datasets:

- **`all_trajs.pkl`** — Trajectory data operates on the grid; demographics provide district-level context
- **`active_taxis_*.pkl`** — Active taxi counts can be correlated with district population and employment
- **`pickup_dropoff_counts.pkl`** — Demand patterns can be analyzed against population density and GDP

### Metadata Files

- **`all_demographics_by_district_metadata.csv`** — Column-level metadata (dtypes, uniqueness, sample values)
- **`all_demographics_by_district_metadata.json`** — Machine-readable metadata (dtypes and district list)

---

## Version History

- **Source File:** `raw_data/demographics_by_district.csv` (original Chinese statistical data with English headers)
- **Processing:** Column renaming and metadata generation via `data/demographic_data/create_demographic_metadata.ipynb`
- **Output Files:**
  - `data/demographic_data/all_demographics_by_district.csv` (renamed columns)
  - `data/demographic_data/all_demographics_by_district_metadata.csv`
  - `data/demographic_data/all_demographics_by_district_metadata.json`

---

## Related Resources

- **Processing Notebook:** `data/demographic_data/create_demographic_metadata.ipynb`
- **Raw Source Data:** `raw_data/demographics_by_district.csv`
- **Metadata Files:**
  - `data/demographic_data/all_demographics_by_district_metadata.csv`
  - `data/demographic_data/all_demographics_by_district_metadata.json`
- **Related Datasets:**
  - `all_trajs.pkl` — Expert driver trajectories
  - `pickup_dropoff_counts.pkl` — Pickup and dropoff demand data
  - `active_taxis_*.pkl` — Active taxi count datasets


