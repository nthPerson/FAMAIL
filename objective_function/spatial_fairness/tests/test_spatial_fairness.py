"""
Unit tests for Spatial Fairness term.

Tests cover:
- Gini coefficient computation
- Service rate calculations
- SpatialFairnessTerm class
- Edge cases and validation
"""

import pytest
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spatial_fairness.utils import (
    compute_gini,
    compute_gini_alternative,
    compute_lorenz_curve,
    aggregate_counts_by_period,
    get_unique_periods,
    compute_service_rates_for_period,
    validate_pickup_dropoff_data,
)
from spatial_fairness.config import SpatialFairnessConfig
from spatial_fairness.term import SpatialFairnessTerm


# =============================================================================
# GINI COEFFICIENT TESTS
# =============================================================================

class TestGiniCoefficient:
    """Test Gini coefficient computation."""
    
    def test_perfect_equality(self):
        """All values equal → Gini = 0."""
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert compute_gini(values) == pytest.approx(0.0, abs=1e-10)
    
    def test_perfect_equality_larger(self):
        """Many equal values → Gini = 0."""
        values = np.array([5.0] * 100)
        assert compute_gini(values) == pytest.approx(0.0, abs=1e-10)
    
    def test_maximum_inequality(self):
        """One value, rest zero → Gini approaches (n-1)/n."""
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        gini = compute_gini(values)
        # For n=5 with one non-zero: Gini = (n-1)/n = 0.8
        assert gini == pytest.approx(0.8, abs=0.01)
    
    def test_moderate_inequality(self):
        """Unequal distribution → 0 < Gini < 1."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gini = compute_gini(values)
        assert 0.0 < gini < 1.0
    
    def test_known_value_uniform_distribution(self):
        """Test against known Gini coefficient for [1,2,3,4,5]."""
        # This distribution has Gini ≈ 0.267
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gini = compute_gini(values)
        assert gini == pytest.approx(0.267, abs=0.01)
    
    def test_empty_array(self):
        """Empty array → Gini = 0."""
        values = np.array([])
        assert compute_gini(values) == 0.0
    
    def test_all_zeros(self):
        """All zeros → Gini = 0 (no inequality among zeros)."""
        values = np.array([0.0, 0.0, 0.0])
        assert compute_gini(values) == 0.0
    
    def test_single_value(self):
        """Single value → Gini = 0."""
        values = np.array([5.0])
        assert compute_gini(values) == 0.0
    
    def test_alternative_matches_primary(self):
        """Alternative formula gives same result."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gini1 = compute_gini(values)
        gini2 = compute_gini_alternative(values)
        assert gini1 == pytest.approx(gini2, abs=0.01)
    
    def test_always_in_valid_range(self):
        """Gini is always in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            values = rng.random(50) * 100
            gini = compute_gini(values)
            assert 0.0 <= gini <= 1.0


class TestLorenzCurve:
    """Test Lorenz curve computation."""
    
    def test_perfect_equality(self):
        """Equal values → 45-degree line."""
        values = np.array([1.0, 1.0, 1.0, 1.0])
        x, y = compute_lorenz_curve(values)
        # Should be close to y = x
        np.testing.assert_allclose(x, y, atol=0.01)
    
    def test_extreme_inequality(self):
        """One rich, rest poor → curve hugs bottom-right."""
        values = np.array([0.0, 0.0, 0.0, 100.0])
        x, y = compute_lorenz_curve(values)
        # Last person has all the wealth, so y[:-1] should be ~0
        assert y[-2] == pytest.approx(0.0, abs=0.01)
        assert y[-1] == 1.0
    
    def test_starts_at_origin(self):
        """Lorenz curve starts at (0, 0)."""
        values = np.array([1.0, 2.0, 3.0])
        x, y = compute_lorenz_curve(values)
        assert x[0] == 0.0
        assert y[0] == 0.0
    
    def test_ends_at_one_one(self):
        """Lorenz curve ends at (1, 1)."""
        values = np.array([1.0, 2.0, 3.0])
        x, y = compute_lorenz_curve(values)
        assert x[-1] == 1.0
        assert y[-1] == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# DATA AGGREGATION TESTS
# =============================================================================

class TestDataAggregation:
    """Test data aggregation functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample pickup/dropoff data."""
        return {
            (1, 1, 1, 1): (10, 5),    # Cell (1,1), time 1, day 1
            (1, 1, 1, 2): (15, 8),    # Cell (1,1), time 1, day 2
            (1, 1, 13, 1): (20, 10),  # Cell (1,1), time 13 (hour 1), day 1
            (2, 2, 1, 1): (5, 2),     # Cell (2,2), time 1, day 1
            (2, 2, 25, 1): (8, 4),    # Cell (2,2), time 25 (hour 2), day 1
        }
    
    def test_aggregate_by_time_bucket(self, sample_data):
        """Aggregate by individual time bucket."""
        pickups, dropoffs = aggregate_counts_by_period(
            sample_data, period_type="time_bucket"
        )
        
        # Check that keys have (cell, (time, day)) format
        assert ((1, 1), (1, 1)) in pickups
        assert pickups[((1, 1), (1, 1))] == 10
    
    def test_aggregate_by_hour(self, sample_data):
        """Aggregate by hour."""
        pickups, dropoffs = aggregate_counts_by_period(
            sample_data, period_type="hourly"
        )
        
        # Time buckets 1-12 → hour 0, 13-24 → hour 1
        # Cell (1,1), hour 0, day 1 should have time bucket 1
        assert ((1, 1), (0, 1)) in pickups
        assert pickups[((1, 1), (0, 1))] == 10  # Only time bucket 1 on day 1
        
        # Hour 1, day 1 should have time bucket 13
        assert ((1, 1), (1, 1)) in pickups
        assert pickups[((1, 1), (1, 1))] == 20
    
    def test_aggregate_by_day(self, sample_data):
        """Aggregate by day."""
        pickups, dropoffs = aggregate_counts_by_period(
            sample_data, period_type="daily"
        )
        
        # Cell (1,1) on day 1 should sum time buckets 1 and 13
        assert ((1, 1), 1) in pickups
        assert pickups[((1, 1), 1)] == 10 + 20  # = 30
    
    def test_aggregate_all(self, sample_data):
        """Aggregate all data."""
        pickups, dropoffs = aggregate_counts_by_period(
            sample_data, period_type="all"
        )
        
        # All data for cell (1,1)
        assert ((1, 1), "all") in pickups
        assert pickups[((1, 1), "all")] == 10 + 15 + 20  # = 45
    
    def test_get_unique_periods(self, sample_data):
        """Get unique periods."""
        pickups, dropoffs = aggregate_counts_by_period(
            sample_data, period_type="daily"
        )
        periods = get_unique_periods(pickups, dropoffs)
        
        assert 1 in periods
        assert 2 in periods


# =============================================================================
# SPATIAL FAIRNESS TERM TESTS
# =============================================================================

class TestSpatialFairnessTerm:
    """Test SpatialFairnessTerm class."""
    
    @pytest.fixture
    def small_config(self):
        """Small grid configuration for testing."""
        return SpatialFairnessConfig(
            grid_dims=(4, 4),
            num_taxis=1,
            num_days=1.0,
            period_type="daily",
            data_is_one_indexed=True,
        )
    
    def test_perfect_equality(self, small_config):
        """Uniform distribution → F_spatial ≈ 1.0."""
        # All cells have same count
        data = {
            (x, y, 1, 1): (10, 10)
            for x in range(1, 5) for y in range(1, 5)
        }
        term = SpatialFairnessTerm(small_config)
        result = term.compute({}, {'pickup_dropoff_counts': data})
        
        # Perfect equality should give F_spatial = 1.0
        assert result == pytest.approx(1.0, abs=0.01)
    
    def test_maximum_inequality(self, small_config):
        """All activity in one cell → F_spatial low."""
        data = {(1, 1, 1, 1): (100, 100)}
        # Add zeros for other cells
        for x in range(1, 5):
            for y in range(1, 5):
                if (x, y) != (1, 1):
                    data[(x, y, 1, 1)] = (0, 0)
        
        term = SpatialFairnessTerm(small_config)
        result = term.compute({}, {'pickup_dropoff_counts': data})
        
        # Should have high inequality (low fairness)
        assert result < 0.3
    
    def test_output_range(self, small_config):
        """Output always in [0, 1]."""
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            data = {
                (x, y, 1, 1): (int(rng.integers(0, 100)), int(rng.integers(0, 100)))
                for x in range(1, 5) for y in range(1, 5)
            }
            term = SpatialFairnessTerm(small_config)
            result = term.compute({}, {'pickup_dropoff_counts': data})
            
            assert 0.0 <= result <= 1.0
    
    def test_determinism(self, small_config):
        """Same input → same output."""
        data = {
            (x, y, 1, 1): (x + y, x * y)
            for x in range(1, 5) for y in range(1, 5)
        }
        term = SpatialFairnessTerm(small_config)
        
        result1 = term.compute({}, {'pickup_dropoff_counts': data})
        result2 = term.compute({}, {'pickup_dropoff_counts': data})
        
        assert result1 == result2
    
    def test_breakdown_structure(self, small_config):
        """Test compute_with_breakdown returns expected structure."""
        data = {
            (x, y, 1, 1): (10, 10)
            for x in range(1, 5) for y in range(1, 5)
        }
        term = SpatialFairnessTerm(small_config)
        breakdown = term.compute_with_breakdown({}, {'pickup_dropoff_counts': data})
        
        # Check required keys
        assert 'value' in breakdown
        assert 'components' in breakdown
        assert 'statistics' in breakdown
        assert 'diagnostics' in breakdown
        
        # Check components
        assert 'avg_gini_arrival' in breakdown['components']
        assert 'avg_gini_departure' in breakdown['components']
        assert 'per_period_data' in breakdown['components']
    
    def test_metadata(self, small_config):
        """Test metadata is correctly populated."""
        term = SpatialFairnessTerm(small_config)
        
        assert term.metadata.name == "spatial_fairness"
        assert term.metadata.higher_is_better is True
        assert term.metadata.value_range == (0.0, 1.0)
    
    def test_missing_data_raises_error(self, small_config):
        """Missing required data raises ValueError."""
        term = SpatialFairnessTerm(small_config)
        
        with pytest.raises(ValueError, match="pickup_dropoff_counts"):
            term.compute({}, {})


class TestDataValidation:
    """Test data validation functions."""
    
    def test_valid_data(self):
        """Valid data passes validation."""
        data = {
            (1, 1, 1, 1): (10, 5),
            (2, 2, 2, 2): (20, 10),
        }
        is_valid, errors = validate_pickup_dropoff_data(data)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_empty_data(self):
        """Empty data fails validation."""
        is_valid, errors = validate_pickup_dropoff_data({})
        
        assert is_valid is False
        assert "empty" in errors[0].lower()
    
    def test_negative_counts(self):
        """Negative counts flagged."""
        data = {
            (1, 1, 1, 1): (-5, 10),  # Negative pickup
        }
        is_valid, errors = validate_pickup_dropoff_data(data)
        
        assert is_valid is False
        assert any("negative pickup" in e.lower() for e in errors)


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================

class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Valid config passes validation."""
        config = SpatialFairnessConfig(
            period_type="hourly",
            grid_dims=(48, 90),
            num_taxis=50,
        )
        config.validate()  # Should not raise
    
    def test_invalid_period_type(self):
        """Invalid period_type raises error."""
        config = SpatialFairnessConfig(period_type="invalid")
        
        with pytest.raises(ValueError, match="period_type"):
            config.validate()
    
    def test_invalid_grid_dims(self):
        """Invalid grid dimensions raises error."""
        config = SpatialFairnessConfig(grid_dims=(0, 90))
        
        with pytest.raises(ValueError, match="Grid"):
            config.validate()
    
    def test_invalid_num_taxis(self):
        """Invalid num_taxis raises error."""
        config = SpatialFairnessConfig(num_taxis=0)
        
        with pytest.raises(ValueError, match="num_taxis"):
            config.validate()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
