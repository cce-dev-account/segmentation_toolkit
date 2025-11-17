"""
Pydantic Models for Segment Information

Type-safe models for segment statistics, profiles, and validation results.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class SegmentStatistics(BaseModel):
    """
    Statistical summary for a single segment.

    Replaces dictionary returns from engine._get_segment_statistics()
    with type-safe validated results.

    Example:
        >>> stats = SegmentStatistics(
        ...     segment_id=1,
        ...     n_observations=5000,
        ...     n_defaults=250,
        ...     default_rate=0.05,
        ...     density=0.25
        ... )
        >>> stats.default_rate
        0.05
    """
    segment_id: int = Field(
        ge=0,
        description="Unique identifier for this segment"
    )

    n_observations: int = Field(
        ge=1,
        description="Total number of observations in segment"
    )

    n_defaults: int = Field(
        ge=0,
        description="Number of default events in segment"
    )

    default_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion of defaults (PD estimate)"
    )

    density: float = Field(
        gt=0.0,
        le=1.0,
        description="Proportion of total population in this segment"
    )

    @model_validator(mode='after')
    def validate_defaults_not_exceed_observations(self):
        """Ensure n_defaults <= n_observations."""
        if self.n_defaults > self.n_observations:
            raise ValueError(
                f"n_defaults ({self.n_defaults}) cannot exceed "
                f"n_observations ({self.n_observations})"
            )
        return self

    @model_validator(mode='after')
    def validate_default_rate_consistency(self):
        """Ensure default_rate matches n_defaults/n_observations."""
        expected_rate = self.n_defaults / self.n_observations
        # Allow small floating point differences
        if abs(expected_rate - self.default_rate) > 1e-6:
            raise ValueError(
                f"default_rate ({self.default_rate:.6f}) inconsistent with "
                f"n_defaults/n_observations ({expected_rate:.6f})"
            )
        return self

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Segment {self.segment_id}: "
            f"{self.n_observations:,} obs, "
            f"{self.n_defaults} defaults, "
            f"PD={self.default_rate:.4f}, "
            f"density={self.density:.2%}"
        )


class FeatureProfile(BaseModel):
    """
    Statistical profile of a feature within a segment.

    Used to characterize segment composition and discriminative power.
    """
    feature_name: str = Field(description="Name of the feature")

    median: float = Field(description="Median value in segment")
    mean: float = Field(description="Mean value in segment")
    std: float = Field(ge=0.0, description="Standard deviation in segment")

    p5: float = Field(description="5th percentile")
    p95: float = Field(description="95th percentile")

    global_median: Optional[float] = Field(
        default=None,
        description="Median value in overall population (for comparison)"
    )

    relative_difference: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Relative difference from global median (discriminative power)"
    )


class TreeRuleType(str, Enum):
    """Type of tree rule for segment definition."""
    SINGLE = "single"  # Single path from tree root to leaf
    MULTIPLE = "multiple"  # Multiple paths merged into segment


class TreeRule(BaseModel):
    """
    Decision tree rule(s) defining a segment.

    Can represent either a single path or multiple merged paths.
    """
    rule_type: TreeRuleType = Field(description="Type of rule")

    # For single path rules
    rule_string: Optional[str] = Field(
        default=None,
        description="IF-THEN rule string (for single path)"
    )

    # For multiple path rules
    n_paths: Optional[int] = Field(
        default=None,
        ge=2,
        description="Number of tree paths merged (for multiple paths)"
    )

    paths: Optional[List[str]] = Field(
        default=None,
        description="List of individual path rules (for multiple paths)"
    )

    leaf_nodes: Optional[List[int]] = Field(
        default=None,
        description="Tree leaf node IDs in this segment"
    )

    @model_validator(mode='after')
    def validate_rule_consistency(self):
        """Ensure rule fields match rule_type."""
        if self.rule_type == TreeRuleType.SINGLE:
            if not self.rule_string:
                raise ValueError("SINGLE rule must have rule_string")
        elif self.rule_type == TreeRuleType.MULTIPLE:
            if not self.paths or not self.n_paths:
                raise ValueError("MULTIPLE rule must have paths and n_paths")
            if len(self.paths) != self.n_paths:
                raise ValueError(
                    f"n_paths ({self.n_paths}) must match "
                    f"length of paths ({len(self.paths)})"
                )
        return self


class AdjustmentType(str, Enum):
    """Types of adjustments applied to segments."""
    MERGE = "merge"
    SPLIT = "split"
    FORCED_SPLIT = "forced_split"
    MONOTONICITY_FIX = "monotonicity_fix"


class SegmentAdjustmentHistory(BaseModel):
    """
    Record of all adjustments applied to create/modify a segment.

    Tracks how segments were created through merges, splits, and other
    post-tree processing operations.
    """
    created_by_merges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of merge operations that created this segment"
    )

    created_by_splits: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of split operations that created this segment"
    )

    created_by_forced_splits: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of forced split operations that created this segment"
    )

    involved_in_merges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of merge operations involving this segment"
    )

    has_monotonicity_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of monotonicity violations for this segment"
    )

    def has_adjustments(self) -> bool:
        """Check if any adjustments were applied."""
        return any([
            self.created_by_merges,
            self.created_by_splits,
            self.created_by_forced_splits,
            self.involved_in_merges,
            self.has_monotonicity_violations
        ])

    def get_summary(self) -> str:
        """Get human-readable summary of adjustments."""
        if not self.has_adjustments():
            return "No adjustments applied (original tree segment)"

        parts = []
        if self.created_by_merges:
            parts.append(f"{len(self.created_by_merges)} merges")
        if self.created_by_splits:
            parts.append(f"{len(self.created_by_splits)} splits")
        if self.created_by_forced_splits:
            parts.append(f"{len(self.created_by_forced_splits)} forced splits")
        if self.has_monotonicity_violations:
            parts.append(f"{len(self.has_monotonicity_violations)} monotonicity violations")

        return "Adjustments: " + ", ".join(parts)


class ComprehensiveSegmentDescription(BaseModel):
    """
    Complete description of a segment including statistics, rules, and profile.

    Replaces dictionary returns from engine.get_comprehensive_segment_descriptions()
    with type-safe validated results.

    Example:
        >>> desc = ComprehensiveSegmentDescription(
        ...     segment_id=1,
        ...     statistics=SegmentStatistics(...),
        ...     tree_rule=TreeRule(rule_type='single', rule_string='IF age > 30 ...'),
        ...     feature_profile={...},
        ...     adjustment_history=SegmentAdjustmentHistory(...)
        ... )
    """
    segment_id: int = Field(
        ge=0,
        description="Unique identifier for this segment"
    )

    statistics: SegmentStatistics = Field(
        description="Statistical summary of segment"
    )

    tree_rule: Optional[TreeRule] = Field(
        default=None,
        description="Decision tree rule(s) defining this segment"
    )

    feature_profile: Dict[str, FeatureProfile] = Field(
        default_factory=dict,
        description="Statistical profile for each feature in segment"
    )

    top_discriminative_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Features with highest discriminative power for this segment"
    )

    adjustment_history: SegmentAdjustmentHistory = Field(
        default_factory=SegmentAdjustmentHistory,
        description="Record of adjustments applied to this segment"
    )

    @model_validator(mode='after')
    def validate_segment_id_consistency(self):
        """Ensure segment_id matches statistics.segment_id."""
        if self.statistics.segment_id != self.segment_id:
            raise ValueError(
                f"segment_id ({self.segment_id}) must match "
                f"statistics.segment_id ({self.statistics.segment_id})"
            )
        return self


class SegmentValidation(BaseModel):
    """
    Validation results specific to a segment.

    Used to track which segments pass/fail specific validation criteria.
    """
    segment_id: int = Field(
        ge=0,
        description="Segment being validated"
    )

    passes_density_requirement: bool = Field(
        description="Meets minimum/maximum density constraints"
    )

    passes_defaults_requirement: bool = Field(
        description="Has minimum required defaults"
    )

    passes_differentiation_requirement: bool = Field(
        description="Sufficiently different PD from other segments"
    )

    passes_all: bool = Field(
        description="Passes all validation requirements"
    )

    violations: List[str] = Field(
        default_factory=list,
        description="List of validation requirements that failed"
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings for this segment"
    )

    @model_validator(mode='after')
    def validate_passes_all_consistency(self):
        """Ensure passes_all matches individual pass flags."""
        expected_passes_all = (
            self.passes_density_requirement and
            self.passes_defaults_requirement and
            self.passes_differentiation_requirement
        )

        if self.passes_all != expected_passes_all:
            raise ValueError(
                f"passes_all ({self.passes_all}) inconsistent with individual checks "
                f"(density={self.passes_density_requirement}, "
                f"defaults={self.passes_defaults_requirement}, "
                f"diff={self.passes_differentiation_requirement})"
            )

        return self

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.passes_all:
            return f"Segment {self.segment_id}: [PASS] All validations passed"

        return (
            f"Segment {self.segment_id}: [FAIL] "
            f"{len(self.violations)} violation(s): {', '.join(self.violations)}"
        )
