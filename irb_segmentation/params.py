"""
IRB Segmentation Parameters Configuration

This module defines the IRBSegmentationParams dataclass that captures all
segmentation requirements for IRB PD models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import warnings


@dataclass
class IRBSegmentationParams:
    """
    Configuration parameters for IRB PD model segmentation.

    This class captures both sklearn-compatible parameters and IRB-specific
    regulatory/business requirements.

    Attributes:
        # Sklearn-compatible parameters
        max_depth: Maximum depth of the decision tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        criterion: Split quality measure ('gini' or 'entropy')
        random_state: Random seed for reproducibility

        # IRB statistical requirements
        min_defaults_per_leaf: Minimum default events per segment (Basel regulatory)
        min_default_rate_diff: Minimum PD difference between segments
        significance_level: Statistical significance level for tests

        # Segment density controls
        min_segment_density: Minimum proportion of observations per segment
        max_segment_density: Maximum proportion of observations per segment

        # Business constraints
        monotone_constraints: Feature monotonicity requirements
        forced_splits: Mandatory split points for specific features

        # Validation configuration
        validation_tests: List of validation tests to run
    """

    # Sklearn-compatible params
    max_depth: int = 3
    min_samples_split: int = 1000
    min_samples_leaf: int = 500
    criterion: str = 'gini'
    random_state: Optional[int] = 42

    # IRB statistical requirements (beyond sklearn)
    min_defaults_per_leaf: int = 20  # Basel regulatory minimum
    min_default_rate_diff: float = 0.001  # Minimum PD difference between segments
    significance_level: float = 0.01  # For statistical tests

    # Segment density controls
    min_segment_density: float = 0.10  # Min 10% of observations per segment
    max_segment_density: float = 0.50  # Max 50% of observations per segment

    # Business constraints
    monotone_constraints: Dict[str, int] = field(default_factory=dict)
    forced_splits: Dict[str, Union[float, List[str]]] = field(default_factory=dict)  # Numeric threshold or categorical values

    # Validation tests to run
    validation_tests: List[str] = field(
        default_factory=lambda: ['chi_squared', 'psi', 'binomial']
    )

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        issues = self.validate_params()
        if issues:
            raise ValueError(f"Invalid parameters:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def to_sklearn_params(self) -> Dict[str, Union[int, str, None]]:
        """
        Extract only sklearn-compatible parameters.

        Returns:
            Dictionary of parameters that can be passed to sklearn DecisionTreeClassifier
        """
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'random_state': self.random_state,
        }

    def validate_params(self) -> List[str]:
        """
        Check parameter consistency and return any issues.

        Returns:
            List of validation error messages (empty if all valid)
        """
        issues = []

        # Check sklearn parameters
        if self.max_depth <= 0:
            issues.append("max_depth must be positive")

        if self.min_samples_split < 2:
            issues.append("min_samples_split must be at least 2")

        if self.min_samples_leaf < 1:
            issues.append("min_samples_leaf must be at least 1")

        if self.min_samples_split < 2 * self.min_samples_leaf:
            issues.append(
                f"min_samples_split ({self.min_samples_split}) should be at least "
                f"2 * min_samples_leaf ({2 * self.min_samples_leaf})"
            )

        if self.criterion not in ['gini', 'entropy']:
            issues.append(f"criterion must be 'gini' or 'entropy', got '{self.criterion}'")

        # Check IRB requirements
        if self.min_defaults_per_leaf < 1:
            issues.append("min_defaults_per_leaf must be at least 1")

        if self.min_defaults_per_leaf < 20:
            warnings.warn(
                f"min_defaults_per_leaf={self.min_defaults_per_leaf} is below Basel "
                f"recommended minimum of 20 defaults per segment"
            )

        if self.min_default_rate_diff < 0:
            issues.append("min_default_rate_diff must be non-negative")

        if not (0 < self.significance_level < 1):
            issues.append("significance_level must be between 0 and 1")

        # Check density constraints
        if not (0 < self.min_segment_density < 1):
            issues.append("min_segment_density must be between 0 and 1")

        if not (0 < self.max_segment_density <= 1):
            issues.append("max_segment_density must be between 0 and 1")

        if self.min_segment_density >= self.max_segment_density:
            issues.append(
                f"min_segment_density ({self.min_segment_density}) must be less than "
                f"max_segment_density ({self.max_segment_density})"
            )

        # Check business constraints
        for feature, direction in self.monotone_constraints.items():
            if direction not in [-1, 0, 1]:
                issues.append(
                    f"monotone_constraints for '{feature}' must be -1, 0, or 1, got {direction}"
                )

        # Check validation tests
        valid_tests = {'chi_squared', 'psi', 'binomial', 'ks', 'gini'}
        invalid_tests = set(self.validation_tests) - valid_tests
        if invalid_tests:
            issues.append(
                f"Invalid validation tests: {invalid_tests}. "
                f"Valid tests are: {valid_tests}"
            )

        return issues

    def get_summary(self) -> str:
        """
        Get a human-readable summary of the configuration.

        Returns:
            Formatted string describing the configuration
        """
        lines = [
            "IRB Segmentation Parameters",
            "=" * 50,
            "",
            "Sklearn Parameters:",
            f"  max_depth: {self.max_depth}",
            f"  min_samples_split: {self.min_samples_split}",
            f"  min_samples_leaf: {self.min_samples_leaf}",
            f"  criterion: {self.criterion}",
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
            f"  forced_splits: {self.forced_splits or 'None'}",
            "",
            "Validation Tests:",
            f"  {', '.join(self.validation_tests)}",
        ]
        return "\n".join(lines)
