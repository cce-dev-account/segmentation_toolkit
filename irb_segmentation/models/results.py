"""
Pydantic Models for IRB Segmentation Results

Type-safe result models replacing dictionary returns with validated objects.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import numpy as np


class ValidationTestResult(BaseModel):
    """
    Results from a single validation test.

    Generic structure that accommodates different validation test types
    (chi-squared, PSI, binomial, KS, density, defaults, etc.)
    """
    test_name: str = Field(description="Name of the validation test")
    passed: bool = Field(description="Whether the test passed")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Test-specific results and metrics"
    )
    failed_segments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of segments that failed this test"
    )

    class Config:
        arbitrary_types_allowed = True


class ValidationResult(BaseModel):
    """
    Comprehensive validation results from all configured tests.

    This model replaces dictionary returns from SegmentValidator.run_all_validations
    with type-safe, validated results.

    Example:
        >>> validation = ValidationResult(
        ...     all_passed=True,
        ...     validations={
        ...         'chi_squared': ValidationTestResult(
        ...             test_name='chi_squared',
        ...             passed=True,
        ...             details={'p_value': 0.001, 'statistic': 42.5}
        ...         ),
        ...         'density': ValidationTestResult(
        ...             test_name='density',
        ...             passed=True,
        ...             details={'min_density': 0.10, 'max_density': 0.50}
        ...         )
        ...     }
        ... )
    """
    all_passed: bool = Field(description="True if all validation tests passed")

    validations: Dict[str, ValidationTestResult] = Field(
        default_factory=dict,
        description="Results for each validation test run"
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings from validation"
    )

    def get_failed_tests(self) -> List[str]:
        """Get list of test names that failed."""
        return [
            name for name, result in self.validations.items()
            if not result.passed
        ]

    def get_summary(self) -> str:
        """Get human-readable summary of validation results."""
        if self.all_passed:
            return f"[PASS] All {len(self.validations)} validation tests passed"

        failed = self.get_failed_tests()
        return (
            f"[FAIL] {len(failed)}/{len(self.validations)} tests failed: "
            f"{', '.join(failed)}"
        )


class SegmentationResult(BaseModel):
    """
    Complete results from segmentation model fitting.

    This model replaces dictionary returns from engine.get_validation_report()
    with type-safe results. Contains all information needed for model deployment
    and regulatory reporting.

    Example:
        >>> result = SegmentationResult(
        ...     timestamp=datetime.now(),
        ...     n_segments=5,
        ...     train_size=10000,
        ...     validation_size=5000,
        ...     train_validation=ValidationResult(all_passed=True, validations={}),
        ...     val_validation=ValidationResult(all_passed=True, validations={}),
        ...     segment_statistics={0: SegmentStatistics(...), 1: ...},
        ...     params_used=IRBSegmentationParams(...)
        ... )
    """
    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the segmentation was created"
    )

    # Segment counts
    n_segments: int = Field(
        ge=1,
        description="Number of final segments after adjustments"
    )

    train_size: int = Field(
        ge=1,
        description="Number of training observations"
    )

    validation_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of validation observations (if provided)"
    )

    # Validation results
    train_validation: ValidationResult = Field(
        description="Validation results on training data"
    )

    val_validation: Optional[ValidationResult] = Field(
        default=None,
        description="Validation results on validation data (if provided)"
    )

    # Segment statistics (detailed stats are in SegmentStatistics models)
    segment_statistics: Dict[int, 'SegmentStatistics'] = Field(
        description="Statistics for each segment"
    )

    # Adjustment history
    adjustment_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each adjustment type applied"
    )

    # Parameters used
    params_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key parameters used for this segmentation"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist()
        }

    @model_validator(mode='after')
    def validate_segment_count(self):
        """Ensure segment_statistics matches n_segments."""
        if len(self.segment_statistics) != self.n_segments:
            raise ValueError(
                f"segment_statistics has {len(self.segment_statistics)} entries "
                f"but n_segments={self.n_segments}"
            )
        return self

    def is_production_ready(self) -> bool:
        """
        Check if segmentation is ready for production deployment.

        Returns:
            True if all validations passed on both train and val sets
        """
        train_ok = self.train_validation.all_passed
        val_ok = (
            self.val_validation.all_passed
            if self.val_validation is not None
            else True
        )
        return train_ok and val_ok

    def get_summary(self) -> str:
        """
        Get human-readable summary of segmentation results.

        Returns:
            Formatted string with key metrics
        """
        lines = [
            "Segmentation Results",
            "=" * 60,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Segments: {self.n_segments}",
            f"Training observations: {self.train_size:,}",
        ]

        if self.validation_size:
            lines.append(f"Validation observations: {self.validation_size:,}")

        lines.extend([
            "",
            "Training Validation:",
            f"  {self.train_validation.get_summary()}",
        ])

        if self.val_validation:
            lines.extend([
                "",
                "Validation Set Validation:",
                f"  {self.val_validation.get_summary()}",
            ])

        if self.adjustment_summary:
            lines.extend([
                "",
                "Adjustments Applied:",
            ])
            for adj_type, count in self.adjustment_summary.items():
                lines.append(f"  {adj_type}: {count}")

        lines.extend([
            "",
            f"Production Ready: {'Yes' if self.is_production_ready() else 'No'}",
        ])

        return "\n".join(lines)


# Forward reference resolution
from .segment_info import SegmentStatistics
SegmentationResult.update_forward_refs()
