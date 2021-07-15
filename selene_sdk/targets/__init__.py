"""
This module contains classes and methods for target feature classes.
These are classes which define a way to access a "target feature" such
as a label or annotation on an input sequence.
"""
from .target import Target
from .genomic_features import GenomicFeatures
from .q_genomic_features import qGenomicFeatures

__all__ = ["Target", "GenomicFeatures", "qGenomicFeatures"]
