"""
Pydantic Models for Segment Adjustment Logs

Type-safe models for tracking segment modifications during IRB constraint enforcement.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Literal
from enum import Enum


class MergeReason(str, Enum):
    """Reasons for merging segments."""
    DENSITY = "density"  # Segment density below minimum
    DEFAULTS = "defaults"  # Insufficient default events


class MergeLog(BaseModel):
    """
    Record of a segment merge operation.

    Tracks when small segments are merged with similar neighbors
    to meet IRB density or minimum defaults requirements.

    Example:
        >>> log = MergeLog(
        ...     merged_segment=3,
        ...     into_segment=1,
        ...     reason=MergeReason.DENSITY,
        ...     original_density=0.05,
        ...     original_defaults=10,
        ...     default_rate_diff=0.002
        ... )
    """
    merged_segment: int = Field(
        ge=0,
        description="Segment that was merged away"
    )

    into_segment: int = Field(
        ge=0,
        description="Target segment that absorbed the merged segment"
    )

    reason: MergeReason = Field(
        description="Reason for merge (density or defaults requirement)"
    )

    original_density: float = Field(
        gt=0.0,
        le=1.0,
        description="Density of merged segment before merge"
    )

    original_defaults: int = Field(
        ge=0,
        description="Number of defaults in merged segment"
    )

    default_rate_diff: float = Field(
        ge=0.0,
        le=1.0,
        description="Absolute difference in default rates between segments"
    )

    @model_validator(mode='after')
    def validate_different_segments(self):
        """Ensure merged_segment and into_segment are different."""
        if self.merged_segment == self.into_segment:
            raise ValueError(
                f"Cannot merge segment into itself "
                f"(merged_segment={self.merged_segment}, "
                f"into_segment={self.into_segment})"
            )
        return self

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Merged segment {self.merged_segment} → {self.into_segment} "
            f"({self.reason.value}: density={self.original_density:.2%}, "
            f"defaults={self.original_defaults})"
        )


class SplitType(str, Enum):
    """Types of split operations."""
    NUMERIC = "numeric"  # Split on numeric threshold
    CATEGORICAL = "categorical"  # Split on categorical values
    DENSITY = "density"  # Split to reduce density


class SplitLog(BaseModel):
    """
    Record of a segment split operation.

    Tracks when large segments are split to meet maximum density constraints
    or improve segment homogeneity.

    Example:
        >>> log = SplitLog(
        ...     split_segment=1,
        ...     new_segment=5,
        ...     feature_idx=2,
        ...     feature_name='credit_score',
        ...     split_type=SplitType.NUMERIC,
        ...     threshold=650.0,
        ...     original_density=0.65
        ... )
    """
    split_segment: Optional[int] = Field(
        default=None,
        ge=0,
        description="Original segment that was split"
    )

    new_segment: int = Field(
        ge=0,
        description="New segment created by split"
    )

    feature_idx: int = Field(
        ge=0,
        description="Index of feature used for split"
    )

    feature_name: str = Field(
        min_length=1,
        description="Name of feature used for split"
    )

    split_type: SplitType = Field(
        description="Type of split operation"
    )

    # For numeric splits
    threshold: Optional[float] = Field(
        default=None,
        description="Split threshold (for numeric splits)"
    )

    # For categorical splits
    categories: Optional[List[str]] = Field(
        default=None,
        description="Categories defining the split (for categorical splits)"
    )

    # Density tracking
    original_density: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Density before split"
    )

    new_densities: Optional[List[float]] = Field(
        default=None,
        description="Densities of resulting segments after split"
    )

    @model_validator(mode='after')
    def validate_split_type_consistency(self):
        """Ensure split type fields match split_type."""
        if self.split_type == SplitType.NUMERIC:
            if self.threshold is None:
                raise ValueError("NUMERIC split must have threshold")
        elif self.split_type == SplitType.CATEGORICAL:
            if not self.categories:
                raise ValueError("CATEGORICAL split must have categories")
        return self

    @field_validator('new_densities')
    @classmethod
    def validate_density_list(cls, v):
        """Ensure densities are valid proportions."""
        if v is not None:
            for density in v:
                if not (0.0 < density <= 1.0):
                    raise ValueError(
                        f"All densities must be in (0, 1], got {density}"
                    )
        return v

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.split_type == SplitType.NUMERIC:
            detail = f"{self.feature_name} @ {self.threshold:.4f}"
        elif self.split_type == SplitType.CATEGORICAL:
            detail = f"{self.feature_name} in {self.categories}"
        else:
            detail = f"{self.feature_name}"

        return (
            f"Split segment {self.split_segment} → {self.new_segment} "
            f"on {detail}"
        )


class ForcedSplitLog(BaseModel):
    """
    Record of a forced split operation.

    Tracks splits mandated by business rules rather than statistical optimization.
    Similar to SplitLog but specifically for user-specified forced splits.

    Example:
        >>> log = ForcedSplitLog(
        ...     segment=1,
        ...     new_segment=6,
        ...     feature_idx=0,
        ...     feature_name='industry_code',
        ...     split_type=SplitType.CATEGORICAL,
        ...     categories=['tech', 'finance'],
        ...     reason='Business requirement: separate tech/finance'
        ... )
    """
    segment: int = Field(
        ge=0,
        description="Original segment that was split"
    )

    new_segment: int = Field(
        ge=0,
        description="New segment created by forced split"
    )

    feature_idx: int = Field(
        ge=0,
        description="Index of feature used for split"
    )

    feature_name: str = Field(
        min_length=1,
        description="Name of feature used for split"
    )

    split_type: SplitType = Field(
        description="Type of split operation"
    )

    # For numeric forced splits
    threshold: Optional[float] = Field(
        default=None,
        description="Split threshold (for numeric splits)"
    )

    # For categorical forced splits
    categories: Optional[List[str]] = Field(
        default=None,
        description="Categories defining the split (for categorical splits)"
    )

    reason: str = Field(
        default="forced_split",
        description="Business reason for forced split"
    )

    @model_validator(mode='after')
    def validate_split_type_consistency(self):
        """Ensure split type fields match split_type."""
        if self.split_type == SplitType.NUMERIC:
            if self.threshold is None:
                raise ValueError("NUMERIC forced split must have threshold")
        elif self.split_type == SplitType.CATEGORICAL:
            if not self.categories:
                raise ValueError("CATEGORICAL forced split must have categories")
        return self

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.split_type == SplitType.NUMERIC:
            detail = f"{self.feature_name} @ {self.threshold:.4f}"
        else:
            detail = f"{self.feature_name} in {self.categories}"

        return f"Forced split segment {self.segment} → {self.new_segment} on {detail}"


class MonotonicityDirection(str, Enum):
    """Direction of monotonicity constraint."""
    INCREASING = "increasing"  # PD should increase with feature
    DECREASING = "decreasing"  # PD should decrease with feature


class MonotonicityViolation(BaseModel):
    """
    Record of a monotonicity constraint violation.

    Tracks segments where the relationship between feature values and
    default rates violates monotonicity constraints.

    Example:
        >>> violation = MonotonicityViolation(
        ...     feature_idx=1,
        ...     feature_name='debt_to_income',
        ...     direction=MonotonicityDirection.INCREASING,
        ...     segment1=2,
        ...     segment2=3,
        ...     feature_median1=0.30,
        ...     feature_median2=0.45,
        ...     default_rate1=0.08,
        ...     default_rate2=0.06,
        ...     violated=True
        ... )
    """
    feature_idx: int = Field(
        ge=0,
        description="Index of feature with monotonicity constraint"
    )

    feature_name: str = Field(
        min_length=1,
        description="Name of feature with monotonicity constraint"
    )

    direction: MonotonicityDirection = Field(
        description="Expected monotonicity direction"
    )

    segment1: int = Field(
        ge=0,
        description="First segment in comparison"
    )

    segment2: int = Field(
        ge=0,
        description="Second segment in comparison"
    )

    feature_median1: float = Field(
        description="Median feature value in segment1"
    )

    feature_median2: float = Field(
        description="Median feature value in segment2"
    )

    default_rate1: float = Field(
        ge=0.0,
        le=1.0,
        description="Default rate in segment1"
    )

    default_rate2: float = Field(
        ge=0.0,
        le=1.0,
        description="Default rate in segment2"
    )

    violated: bool = Field(
        description="Whether monotonicity was violated"
    )

    @model_validator(mode='after')
    def validate_different_segments(self):
        """Ensure segment1 and segment2 are different."""
        if self.segment1 == self.segment2:
            raise ValueError(
                f"Cannot compare segment to itself "
                f"(segment1={self.segment1}, segment2={self.segment2})"
            )
        return self

    @model_validator(mode='after')
    def validate_violation_logic(self):
        """Verify that violation flag matches the actual violation."""
        # Determine actual violation
        if self.feature_median2 > self.feature_median1:  # segment2 has higher feature value
            if self.direction == MonotonicityDirection.INCREASING:
                actual_violation = self.default_rate2 < self.default_rate1  # PD should increase but decreased
            else:  # DECREASING
                actual_violation = self.default_rate2 > self.default_rate1  # PD should decrease but increased

            # Only validate if we can determine the violation
            if actual_violation != self.violated:
                raise ValueError(
                    f"Violation flag ({self.violated}) inconsistent with actual data: "
                    f"direction={self.direction.value}, med1={self.feature_median1}, "
                    f"med2={self.feature_median2}, rate1={self.default_rate1}, "
                    f"rate2={self.default_rate2}"
                )

        return self

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if not self.violated:
            return (
                f"[OK] {self.feature_name} monotonicity ({self.direction.value}) "
                f"satisfied between segments {self.segment1} and {self.segment2}"
            )

        return (
            f"[VIOLATION] {self.feature_name} should be {self.direction.value} "
            f"but segment {self.segment1} (med={self.feature_median1:.4f}, "
            f"PD={self.default_rate1:.4f}) vs segment {self.segment2} "
            f"(med={self.feature_median2:.4f}, PD={self.default_rate2:.4f})"
        )


class AdjustmentHistory(BaseModel):
    """
    Complete history of all adjustments applied during segmentation.

    Replaces dictionary-based adjustment_log_ with type-safe structure.

    Example:
        >>> history = AdjustmentHistory(
        ...     merges=[MergeLog(...), MergeLog(...)],
        ...     splits=[SplitLog(...)],
        ...     forced_splits=[ForcedSplitLog(...)],
        ...     monotonicity_violations=[]
        ... )
        >>> history.total_adjustments()
        3
    """
    merges: List[MergeLog] = Field(
        default_factory=list,
        description="List of merge operations performed"
    )

    splits: List[SplitLog] = Field(
        default_factory=list,
        description="List of split operations performed"
    )

    forced_splits: List[ForcedSplitLog] = Field(
        default_factory=list,
        description="List of forced split operations performed"
    )

    monotonicity_violations: List[MonotonicityViolation] = Field(
        default_factory=list,
        description="List of monotonicity violations detected"
    )

    def total_adjustments(self) -> int:
        """Get total number of adjustments (excluding violations)."""
        return len(self.merges) + len(self.splits) + len(self.forced_splits)

    def has_adjustments(self) -> bool:
        """Check if any adjustments were made."""
        return self.total_adjustments() > 0

    def has_violations(self) -> bool:
        """Check if any monotonicity violations exist."""
        return any(v.violated for v in self.monotonicity_violations)

    def get_summary(self) -> str:
        """Get human-readable summary of all adjustments."""
        lines = [
            "Adjustment History",
            "=" * 50,
            f"Merges: {len(self.merges)}",
            f"Splits: {len(self.splits)}",
            f"Forced splits: {len(self.forced_splits)}",
            f"Monotonicity violations: {sum(1 for v in self.monotonicity_violations if v.violated)}/{len(self.monotonicity_violations)} checked",
        ]

        if not self.has_adjustments():
            lines.append("\nNo adjustments were necessary.")

        return "\n".join(lines)

    def to_legacy_dict(self) -> dict:
        """
        Convert to legacy dictionary format for backward compatibility.

        Returns:
            Dictionary matching old adjustment_log_ structure
        """
        return {
            'merges': [m.dict() for m in self.merges],
            'splits': [s.dict() for s in self.splits],
            'forced_splits': [f.dict() for f in self.forced_splits],
            'monotonicity_violations': [v.dict() for v in self.monotonicity_violations]
        }

    @classmethod
    def from_legacy_dict(cls, log_dict: dict) -> 'AdjustmentHistory':
        """
        Create from legacy dictionary format.

        Args:
            log_dict: Old-style adjustment_log_ dictionary

        Returns:
            AdjustmentHistory instance
        """
        return cls(
            merges=[MergeLog(**m) for m in log_dict.get('merges', [])],
            splits=[SplitLog(**s) for s in log_dict.get('splits', [])],
            forced_splits=[ForcedSplitLog(**f) for f in log_dict.get('forced_splits', [])],
            monotonicity_violations=[
                MonotonicityViolation(**v)
                for v in log_dict.get('monotonicity_violations', [])
            ]
        )
