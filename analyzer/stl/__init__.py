"""
Signal Temporal Logic (STL) integration for TransInferSim.

This package provides STL-based temporal constraint specification and monitoring
for hardware design-space exploration of Transformer accelerators.

Key Components:
- Monitors: Offline and online STL monitoring
- Specifications: STL formula definitions and evaluation
- Signals: Time-series signal extraction from simulation statistics
- Constraints: Pre-defined constraint libraries (performance, power, resources)
- DSE: Design space exploration with STL-guided filtering and ranking
"""

# Check for STL library availability
try:
    import rtamt
    STL_AVAILABLE = True
    STL_BACKEND = 'rtamt'
except ImportError:
    STL_AVAILABLE = False
    STL_BACKEND = None
    import warnings
    warnings.warn(
        "rtamt library not installed. STL features will have limited functionality. "
        "Install with: pip install rtamt",
        ImportWarning
    )

# Core exports
from .monitors.offline_monitor import OfflineSTLMonitor
from .core.specification import STLSpecification
from .signals.signal_extractor import SignalExtractor

# Constraint libraries
from .constraints.performance_constraints import PerformanceConstraints
from .constraints.power_constraints import PowerConstraints
from .constraints.resource_constraints import ResourceConstraints

# DSE utilities
from .dse.constraint_checker import ConstraintBasedDSE

__all__ = [
    'STL_AVAILABLE',
    'STL_BACKEND',
    'OfflineSTLMonitor',
    'STLSpecification',
    'SignalExtractor',
    'PerformanceConstraints',
    'PowerConstraints',
    'ResourceConstraints',
    'ConstraintBasedDSE',
]

__version__ = '0.1.0'
