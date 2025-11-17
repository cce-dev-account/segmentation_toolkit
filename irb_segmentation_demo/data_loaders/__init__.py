"""
Data Loaders for IRB Segmentation Framework

Loaders for public credit/lending datasets commonly used for testing
PD models and credit risk segmentation.
"""

from .base import BaseDataLoader
from .german_credit import GermanCreditLoader, load_german_credit
from .taiwan_credit import TaiwanCreditLoader, load_taiwan_credit
from .lending_club import LendingClubLoader, load_lending_club
from .home_credit import HomeCreditLoader, load_home_credit

__all__ = [
    "BaseDataLoader",
    "GermanCreditLoader",
    "TaiwanCreditLoader",
    "LendingClubLoader",
    "HomeCreditLoader",
    "load_german_credit",
    "load_taiwan_credit",
    "load_lending_club",
    "load_home_credit",
]
