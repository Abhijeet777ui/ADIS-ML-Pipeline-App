"""
Custom exceptions for the ADIS pipeline.
"""

class ADISError(Exception):
    """Base exception for all ADIS errors."""
    pass

class DataLeakageError(ADISError):
    """Raised when severe data leakage is detected (e.g., a feature perfectly predicts the target)."""
    pass

class TargetColumnMissingError(ADISError):
    """Raised when the specified target column is not found in the dataset."""
    pass

class EmptyDatasetError(ADISError):
    """Raised when the dataset is completely empty after filtering or loading."""
    pass

class SchemaValidationError(ADISError):
    """Raised when the dataset fails critical schema validation during ingestion."""
    pass

class ConvergenceFailedError(ADISError):
    """Raised when the AutoResearch agent fails to converge on a better model after max iterations."""
    pass
