# Study Area and Data Sources

## Study Area

The FAMAIL framework operates on real-world GPS taxi trajectory data from **Shenzhen, China** — one of the largest and most economically dynamic cities in southern China. Shenzhen's taxi fleet is one of the densest in the world, and the city's rapid economic growth has produced significant variation in neighborhood wealth and service access — making it an ideal setting for studying fairness in urban transportation.

---

## Spatial and Temporal Resolution

| Parameter | Value |
|-----------|-------|
| **Geographic area** | Shenzhen, China |
| **Spatial grid** | 48 × 90 cells (~1.1 km per cell) |
| **Total grid cells** | 4,320 |
| **Temporal resolution** | 288 time buckets per day (5-minute intervals) |
| **Analysis period** | July–September 2016 (weekdays only) |
| **Fleet** | 50 expert taxi drivers (subset of 17,877 total) |

---

## Data Sources

The project uses three primary data sources, all derived from GPS traces of the 50 expert drivers:

### 1. Taxi Trajectories

GPS-derived trajectory sequences recording each driver's path from the start of passenger-seeking behavior to the pickup event. Each trajectory is a sequence of spatiotemporal states — grid cell coordinates and time-of-day — capturing the driver's movement through the city.

### 2. Pickup and Dropoff Counts

Aggregated event data recording how many passenger pickups and dropoffs occurred at each grid cell, in each 5-minute time window, on each day. This serves as the empirical measure of taxi service distribution across the city.

### 3. Active Taxi Counts

For each grid cell and time period, how many unique taxis were present in the surrounding neighborhood. This measures taxi supply — how many drivers were available to serve each area — and is essential for computing service rates that account for taxi availability.

---

## Demographic Data

District-level demographic data for Shenzhen's 10 administrative districts provides the basis for measuring whether service inequality correlates with socioeconomic factors. Key indicators include:

- GDP per capita
- Average housing price
- Employee compensation per capita
- Population density

Each grid cell is mapped to its majority-overlap Shenzhen district via ArcGIS spatial analysis, allowing demographic attributes to be associated with service patterns at the grid level.

!!! info "Why normalize by active taxis?"
    Raw pickup counts can be misleading — a cell might have few pickups simply because few taxis pass through it. By dividing by the number of active taxis, we obtain a **service rate** that measures how effectively taxis serve an area given their presence. This is the quantity that the [spatial fairness](../fairness.md) metric operates on.

---

<div style="display: flex; justify-content: space-between; margin-top: 2rem;">
<a href="../methodology/discriminator/" class="md-button">← ST-SiameseNet</a>
<a href="../fairness/" class="md-button md-button--primary">Next: Fairness Definitions →</a>
</div>
