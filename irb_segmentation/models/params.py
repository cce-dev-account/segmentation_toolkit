"""
Pydantic Models for IRB Segmentation Parameters

Type-safe, validated configuration parameters with cross-field validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Literal
from enum import Enum
import warnings


class SplitCriterion(str, Enum):
    """Allowed split criteria for decision trees."""
    GINI = "gini"
    ENTROPY = "entropy"


class MonotonicityDirection(int, Enum):
    """Monotonicity constraint directions."""
    DECREASING = -1
    NONE = 0
    INCREASING = 1


class ValidationTest(str, Enum):
    """Available validation tests."""
    CHI_SQUARED = "chi_squared"
    PSI = "psi"
    BINOMIAL = "binomial"
    KS = "ks"
    GINI = "gini"


class IRBSegmentationParams(BaseModel):
    """
    Type-safe configuration parameters for IRB PD model segmentation.

    This model provides:
    - Automatic validation of all parameters
    - Cross-field validation for logical consistency
    - Immutability after creation
    - Easy serialization to JSON/YAML
    - Clear type constraints

    Example:
        >>> params = IRBSegmentationParams(
        ...     max_depth=3,
        ...     min_defaults_per_leaf=30,
        ...     min_segment_density=0.10
        ... )
        >>> params.max_depth
        3
        >>> params.max_depth = 5  # Raises error - immutable
    """

    # ===== Sklearn-compatible parameters =====
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum depth of the decision tree"
    )

    min_samples_split: int = Field(
        default=1000,
        ge=2,
        description="Minimum samples required to split a node"
    )

    min_samples_leaf: int = Field(
        default=500,
        ge=1,
        description="Minimum samples required in a leaf node"
    )

    criterion: SplitCriterion = Field(
        default=SplitCriterion.GINI,
        description="Split quality measure"
    )

    random_state: Optional[int] = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )

    # ===== IRB statistical requirements =====
    min_defaults_per_leaf: int = Field(
        default=20,
        ge=1,
        description="Minimum default events per segment (Basel regulatory)"
    )

    min_default_rate_diff: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Minimum PD difference between segments"
    )

    significance_level: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Statistical significance level for tests"
    )

    # ===== Segment density controls =====
    min_segment_density: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="Minimum proportion of observations per segment"
    )

    max_segment_density: float = Field(
        default=0.50,
        gt=0.0,
        le=1.0,
        description="Maximum proportion of observations per segment"
    )

    # ===== Business constraints =====
    monotone_constraints: Dict[str, Literal[-1, 0, 1]] = Field(
        default_factory=dict,
        description="Feature monotonicity requirements {feature_name: direction}"
    )

    forced_splits: Dict[str, float | List[str]] = Field(
        default_factory=dict,
        description="Mandatory split points {feature_name: threshold or categories}"
    )

    # ===== Validation configuration =====
    validation_tests: List[ValidationTest] = Field(
        default_factory=lambda: [
            ValidationTest.CHI_SQUARED,
            ValidationTest.PSI,
            ValidationTest.BINOMIAL
        ],
        description="List of validation tests to run"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable after creation
        use_enum_values = True  # Use enum values in dict/JSON
        arbitrary_types_allowed = False  # Strict type checking

    # ===== Validators =====

    @field_validator('max_depth')
    @classmethod
    def validate_max_depth(cls, v):
        """Warn if max_depth is too high for IRB requirements."""
        if v > 5:
            warnings.warn(
                f"max_depth={v} is higher than typical IRB requirements (≤5). "
                "Deeper trees may not meet regulatory interpretability standards.",
                UserWarning
            )
        return v

    @field_validator('min_defaults_per_leaf')
    @classmethod
    def validate_min_defaults(cls, v):
        """Warn if below Basel recommended minimum."""
        if v < 20:
            warnings.warn(
                f"min_defaults_per_leaf={v} is below Basel recommended minimum of 20 "
                "defaults per segment. This may not meet regulatory requirements.",
                UserWarning
            )
        return v

    @field_validator('monotone_constraints')
    @classmethod
    def validate_monotone_constraints(cls, v):
        """Validate monotonicity constraint values."""
        for feature, direction in v.items():
            if direction not in [-1, 0, 1]:
                raise ValueError(
                    f"Monotonicity direction for '{feature}' must be -1, 0, or 1, "
                    f"got {direction}"
                )
        return v

    @model_validator(mode='after')
    def validate_sklearn_consistency(self):
        """Check consistency between sklearn parameters."""
        if self.min_samples_split < 2 * self.min_samples_leaf:
            raise ValueError(
                f"min_samples_split ({self.min_samples_split}) should be at least "
                f"2 * min_samples_leaf ({2 * self.min_samples_leaf}) for balanced splits"
            )
        return self

    @model_validator(mode='after')
    def validate_irb_consistency(self):
        """Check consistency between IRB parameters."""
        if self.min_defaults_per_leaf > self.min_samples_leaf:
            raise ValueError(
                f"min_defaults_per_leaf ({self.min_defaults_per_leaf}) cannot exceed "
                f"min_samples_leaf ({self.min_samples_leaf})"
            )
        return self

    @model_validator(mode='after')
    def validate_density_constraints(self):
        """Check consistency of density constraints."""
        if self.min_segment_density >= self.max_segment_density:
            raise ValueError(
                f"min_segment_density ({self.min_segment_density}) must be less than "
                f"max_segment_density ({self.max_segment_density})"
            )
        return self

    # ===== Helper Methods =====

    def to_sklearn_params(self) -> Dict:
        """
        Extract only sklearn-compatible parameters.

        Returns:
            Dictionary of parameters for sklearn DecisionTreeClassifier
        """
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion.value if isinstance(self.criterion, Enum) else self.criterion,
            'random_state': self.random_state,
        }

    def get_summary(self) -> str:
        """
        Get a human-readable summary of the configuration.

        Returns:
            Formatted string describing the configuration
        """
        criterion_val = self.criterion.value if isinstance(self.criterion, Enum) else self.criterion

        lines = [
            "IRB Segmentation Parameters",
            "=" * 50,
            "",
            "Sklearn Parameters:",
            f"  max_depth: {self.max_depth}",
            f"  min_samples_split: {self.min_samples_split}",
            f"  min_samples_leaf: {self.min_samples_leaf}",
            f"  criterion: {criterion_val}",
            f"  random_state: {self.random_state}",
            "",
            "IRB Requirements:",
            f"  min_defaults_per_leaf: {self.min_defaults_per_leaf}",
            f"  min_default_rate_diff: {self.min_default_rate_diff:.4f}",
            f"  significance_level: {self.significance_level}",
            "",
            "Density Constraints:",
            f"  min_segment_density: {self.min_segment_density:.1%}",
            f"  max_segment_density: {self.max_segment_density:.1%}",
            "",
            "Business Constraints:",
            f"  monotone_constraints: {self.monotone_constraints or 'None'}",
            f"  forced_splits: {len(self.forced_splits)} features" if self.forced_splits else "  forced_splits: None",
            "",
            "Validation Tests:",
            f"  {', '.join(test.value if isinstance(test, Enum) else test for test in self.validation_tests)}",
        ]
        return "\n".join(lines)

    def validate_params(self) -> List[str]:
        """
        Legacy compatibility method - always returns empty list.

        Pydantic validation happens automatically during initialization.
        This method exists for backward compatibility.

        Returns:
            Empty list (validation errors raise exceptions in Pydantic)
        """
        # Pydantic handles all validation automatically
        # This method exists for backward compatibility only
        return []
