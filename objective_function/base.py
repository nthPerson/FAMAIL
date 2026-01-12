"""
Base classes and interfaces for FAMAIL objective function terms.

This module defines the abstract interfaces that all objective function terms
must implement to ensure consistent behavior across the framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time


@dataclass
class TermMetadata:
    """Metadata describing an objective function term."""
    
    name: str                          # e.g., "spatial_fairness"
    display_name: str                  # e.g., "Spatial Fairness"
    version: str                       # Semantic version, e.g., "1.0.0"
    description: str                   # Brief description of what the term measures
    value_range: Tuple[float, float]   # Expected output range, typically (0.0, 1.0)
    higher_is_better: bool             # True for all FAMAIL terms
    is_differentiable: bool            # Whether gradients can be computed
    required_data: List[str]           # List of required dataset names
    optional_data: List[str]           # List of optional dataset names
    author: str                        # Primary developer/researcher
    last_updated: str                  # ISO date of last modification


@dataclass
class TermConfig:
    """
    Base configuration class for objective function terms.
    
    Subclass this for term-specific configuration parameters.
    """
    
    # Common parameters
    enabled: bool = True                    # Whether this term is active
    weight: float = 1.0                     # Weight in objective function (α_i)
    
    # Data source paths
    raw_data_dir: str = "raw_data"          # Directory containing raw GPS data
    source_data_dir: str = "source_data"    # Directory containing processed data
    
    # Computation parameters
    cache_intermediate: bool = True         # Whether to cache intermediate results
    verbose: bool = False                   # Print debug information
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.weight < 0:
            raise ValueError("Weight must be non-negative")


class ObjectiveFunctionTerm(ABC):
    """
    Abstract base class for all FAMAIL objective function terms.
    
    All terms must implement this interface to ensure consistent behavior
    across the objective function framework.
    """
    
    def __init__(self, config: TermConfig):
        """
        Initialize the term with configuration parameters.
        
        Args:
            config: Configuration object containing term-specific parameters
        """
        self.config = config
        self._validate_config()
        self._metadata = self._build_metadata()
        self._cache: Dict[str, Any] = {}
    
    @abstractmethod
    def _build_metadata(self) -> TermMetadata:
        """Build and return the term's metadata."""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """
        Compute the term value for the given trajectories.
        
        Args:
            trajectories: Dictionary mapping driver_id to list of trajectories,
                         where each trajectory is a list of states,
                         and each state is a list of features
            auxiliary_data: Dictionary containing additional datasets
                           (e.g., pickup_dropoff_counts, traffic data)
        
        Returns:
            Term value in [0, 1], where higher values indicate better outcomes
        """
        pass
    
    @abstractmethod
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute the term value with detailed breakdown for debugging/analysis.
        
        Args:
            trajectories: Trajectory data (same as compute())
            auxiliary_data: Additional datasets (same as compute())
        
        Returns:
            Dictionary containing:
                - 'value': float - the final term value
                - 'components': Dict - intermediate values used in computation
                - 'statistics': Dict - summary statistics for analysis
                - 'diagnostics': Dict - debugging information
        """
        pass
    
    def compute_gradient(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Compute the gradient of the term with respect to trajectory modifications.
        
        Default implementation returns None (term is not differentiable).
        Override this method for differentiable terms.
        
        Args:
            trajectories: Trajectory data
            auxiliary_data: Additional datasets
        
        Returns:
            Gradient array or None if not differentiable
        """
        return None
    
    @property
    def metadata(self) -> TermMetadata:
        """Return the term's metadata."""
        return self._metadata
    
    @property
    def name(self) -> str:
        """Return the term's internal name."""
        return self._metadata.name
    
    def validate_input(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the input data meets the term's requirements.
        
        Args:
            trajectories: Trajectory data to validate
            auxiliary_data: Additional datasets to validate
        
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check required data
        for required in self._metadata.required_data:
            if required not in auxiliary_data:
                errors.append(f"Missing required dataset: {required}")
        
        # Check trajectory format
        if trajectories is not None and not isinstance(trajectories, dict):
            errors.append("Trajectories must be a dictionary")
        
        return len(errors) == 0, errors
    
    def clear_cache(self) -> None:
        """Clear any cached intermediate results."""
        self._cache.clear()
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(f"[{self.name}] {message}")
    
    def _time_execution(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time the execution of a function and return (result, elapsed_ms)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms


# Type aliases for clarity
TrajectoryData = Dict[str, List[List[List[float]]]]
AuxiliaryData = Dict[str, Any]
