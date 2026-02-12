"""Matrix utilities for correlation and covariance operations.

Provides the nearest-correlation-matrix projection (Higham's algorithm) and
helper functions for positive-definiteness checking and
covariance/correlation conversion.

Modules
-------
nearest_correlation
    Nearest-correlation-matrix algorithm that projects a symmetric matrix
    onto the positive-semidefinite cone (``nearcorr``).
utilities
    Positive-definiteness checks (``is_pos_def``, ``is_pos_semi_def``) and
    covariance/correlation conversions (``correlation_from_covariance``,
    ``covariance_from_correlation``).
"""

__all__ = [
    "nearest_correlation",
    "utilities",
]
