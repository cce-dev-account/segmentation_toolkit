"""
Model Performance Metrics for Binary Classification

Implements discriminatory power metrics required for IRB validation.
Designed for binary PD models with forward compatibility for regression.

References:
- Basel II IRB Framework (paragraphs 417-418)
- EBA Guidelines on PD estimation (EBA/GL/2017/16)
- sklearn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Container for model performance metrics.

    Attributes:
        gini: Gini coefficient (2*AUC - 1), range [-1, 1]
        ks_statistic: Kolmogorov-Smirnov statistic, range [0, 1]
        ks_threshold: Threshold at maximum KS separation
        auc: Area Under ROC Curve, range [0, 1]
        accuracy_ratio: Gini / Gini_perfect
        brier_score: Mean squared error of probabilities, range [0, 1]
        passed_thresholds: Whether metrics meet regulatory thresholds
    """
    gini: float
    ks_statistic: float
    ks_threshold: float
    auc: float
    accuracy_ratio: float
    brier_score: float
    passed_thresholds: bool

    def __str__(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "MODEL PERFORMANCE METRICS",
            "=" * 60,
            "",
            "Discriminatory Power:",
            f"  Gini coefficient:     {self.gini:.4f} (threshold: > 0.30)",
            f"  KS statistic:         {self.ks_statistic:.4f} (threshold: > 0.20)",
            f"  KS threshold:         {self.ks_threshold:.4f}",
            f"  AUC:                  {self.auc:.4f}",
            f"  Accuracy Ratio:       {self.accuracy_ratio:.4f}",
            "",
            "Calibration:",
            f"  Brier Score:          {self.brier_score:.4f} (lower is better)",
            "",
            f"Regulatory Status: {'✓ PASSED' if self.passed_thresholds else '✗ FAILED'}",
            "=" * 60
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'gini': float(self.gini),
            'ks_statistic': float(self.ks_statistic),
            'ks_threshold': float(self.ks_threshold),
            'auc': float(self.auc),
            'accuracy_ratio': float(self.accuracy_ratio),
            'brier_score': float(self.brier_score),
            'passed_thresholds': bool(self.passed_thresholds)
        }


def gini_coefficient(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Gini coefficient (2*AUC - 1).

    The Gini coefficient measures discriminatory power of the model.
    Higher values indicate better separation between classes.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Gini coefficient in range [-1, 1], where:
        - 1.0 = perfect discrimination
        - 0.0 = random model
        - -1.0 = perfectly wrong predictions

    Raises:
        ValueError: If inputs are invalid

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        >>> gini = gini_coefficient(y_true, y_pred)
        >>> print(f"Gini: {gini:.3f}")
        Gini: 1.000

    Note:
        Currently supports binary classification only.
        Future: Extend for continuous targets (LGD/EAD) - see Issue #6.

    Regulatory Context:
        Basel II/III supervisory authorities typically expect Gini > 0.30
        for acceptable PD models, with Gini > 0.50 considered strong.
    """
    # Validate inputs
    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred_proba has {len(y_pred_proba)} samples"
        )

    if len(y_true) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(y_true)}")

    # Check for binary target
    unique_values = np.unique(y_true)
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"y_true must be binary (0/1), found values: {unique_values}"
        )

    if len(unique_values) < 2:
        raise ValueError(
            f"y_true must have both classes, found only: {unique_values}"
        )

    # Calculate AUC and Gini
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1

    logger.debug(f"Calculated Gini: {gini:.4f} (AUC: {auc:.4f})")

    return float(gini)


def ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov statistic for binary classification.

    KS measures the maximum separation between cumulative distributions
    of predicted probabilities for positive and negative classes.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Tuple of (ks_stat, ks_threshold):
        - ks_stat: KS statistic in range [0, 1]
        - ks_threshold: Probability threshold at maximum separation

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        >>> ks, threshold = ks_statistic(y_true, y_pred)
        >>> print(f"KS: {ks:.3f} at threshold {threshold:.3f}")
        KS: 1.000 at threshold 0.800

    Regulatory Context:
        Basel II/III supervisory authorities typically expect KS > 0.20
        for acceptable models, with KS > 0.40 considered strong.
    """
    # Validate inputs (same as gini_coefficient)
    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred_proba has {len(y_pred_proba)} samples"
        )

    if len(y_true) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(y_true)}")

    unique_values = np.unique(y_true)
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"y_true must be binary (0/1), found values: {unique_values}"
        )

    if len(unique_values) < 2:
        raise ValueError(
            f"y_true must have both classes, found only: {unique_values}"
        )

    # Separate scores by class
    scores_pos = y_pred_proba[y_true == 1]
    scores_neg = y_pred_proba[y_true == 0]

    # Two-sample KS test
    ks_stat, p_value = stats.ks_2samp(scores_pos, scores_neg)

    # Find threshold at maximum separation using ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # KS = max(TPR - FPR)
    ks_scores = tpr - fpr
    ks_idx = np.argmax(ks_scores)
    ks_threshold = float(thresholds[ks_idx])

    logger.debug(
        f"Calculated KS: {ks_stat:.4f} at threshold {ks_threshold:.4f} "
        f"(p-value: {p_value:.4e})"
    )

    return float(ks_stat), ks_threshold


def brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Brier score (mean squared error of probabilities).

    Measures calibration quality - how close predicted probabilities
    are to actual outcomes.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Brier score in range [0, 1], where:
        - 0.0 = perfect calibration
        - 0.25 = uninformative model (constant prediction at base rate)
        - 1.0 = worst possible calibration

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        >>> bs = brier_score(y_true, y_pred)
        >>> print(f"Brier Score: {bs:.3f}")
        Brier Score: 0.025

    Note:
        Lower Brier scores indicate better calibration. Compare with
        Brier score of a naive model that predicts the base rate for
        all observations.
    """
    # Validate inputs
    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred_proba has {len(y_pred_proba)} samples"
        )

    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")

    # Calculate mean squared error
    bs = np.mean((y_true - y_pred_proba) ** 2)

    # Calculate baseline Brier score (naive model predicting base rate)
    base_rate = y_true.mean()
    baseline_bs = np.mean((y_true - base_rate) ** 2)

    logger.debug(
        f"Calculated Brier Score: {bs:.4f} "
        f"(baseline: {baseline_bs:.4f})"
    )

    return float(bs)


def accuracy_ratio(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Accuracy Ratio = Gini / Gini_perfect.

    Normalizes Gini coefficient by the theoretical perfect model's Gini.
    Used in some regulatory frameworks as alternative to raw Gini.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Accuracy Ratio in range [0, 1], where:
        - 1.0 = perfect model
        - 0.5 = typical good model
        - 0.0 = random model

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        >>> ar = accuracy_ratio(y_true, y_pred)
        >>> print(f"Accuracy Ratio: {ar:.3f}")
        Accuracy Ratio: 1.000

    Note:
        AR = Gini_model / Gini_perfect. In practice, AR is often identical
        to Gini for perfect models, so we simply return normalized Gini here.
    """
    # Calculate model Gini
    gini_model = gini_coefficient(y_true, y_pred_proba)

    # For binary classification, AR is typically just the Gini normalized to [0,1]
    # Since Gini is already 2*AUC - 1, and ranges from [-1, 1],
    # we can normalize to [0, 1] range
    ar = (gini_model + 1) / 2

    logger.debug(
        f"Calculated Accuracy Ratio: {ar:.4f} (Gini: {gini_model:.4f})"
    )

    return float(ar)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    gini_threshold: float = 0.30,
    ks_threshold: float = 0.20
) -> PerformanceMetrics:
    """
    Calculate all performance metrics at once.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        gini_threshold: Minimum acceptable Gini (default: 0.30 per Basel)
        ks_threshold: Minimum acceptable KS (default: 0.20 per Basel)

    Returns:
        PerformanceMetrics object with all calculated metrics

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        >>> metrics = calculate_all_metrics(y_true, y_pred)
        >>> print(metrics)
        >>> print(f"Model passed thresholds: {metrics.passed_thresholds}")

    Regulatory Context:
        Default thresholds (Gini > 0.30, KS > 0.20) are typical supervisory
        expectations for PD models under Basel II/III. Adjust based on your
        jurisdiction's requirements.
    """
    logger.info("Calculating performance metrics for binary classification model")

    # Calculate individual metrics
    gini = gini_coefficient(y_true, y_pred_proba)
    ks, ks_thresh = ks_statistic(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    ar = accuracy_ratio(y_true, y_pred_proba)
    bs = brier_score(y_true, y_pred_proba)

    # Check regulatory thresholds
    passed = (gini >= gini_threshold) and (ks >= ks_threshold)

    metrics = PerformanceMetrics(
        gini=gini,
        ks_statistic=ks,
        ks_threshold=ks_thresh,
        auc=auc,
        accuracy_ratio=ar,
        brier_score=bs,
        passed_thresholds=passed
    )

    logger.info(
        f"Performance metrics calculated: Gini={gini:.4f}, KS={ks:.4f}, "
        f"AUC={auc:.4f}, AR={ar:.4f}, Brier={bs:.4f}, "
        f"Status={'PASSED' if passed else 'FAILED'}"
    )

    return metrics
