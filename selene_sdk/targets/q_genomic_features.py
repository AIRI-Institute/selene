import pyBigWig
import numpy as np
from .target import Target

class qGenomicFeatures(Target):
    """
    Stores the dataset specifying sequence regions and features.
    Accepts a path to directory containing bigWig files with feature values

    Parameters
    ----------
    features : list(str)
        non-redundunt fature names
    features_path : list(str)
        locations of coreesponding bigWig files

    Attributes
    ----------
    data : tabix.open
        The data stored in a tabix-indexed `*.bed` file.
    n_features : int
        The number of distinct features.
    """

    def __init__(self, features, features_path):
        """
        Constructs a new `GenomicFeatures` object.
        """

        self.features =  features
        self._feature_handlers = {i: pyBigWig.open(j) \
                                            for i,j in zip(features,features_path)
                                      }
        self.n_features = len(features)


    def get_feature_data(self, chrom, start, end):
        """
        For a sequence of length :math:`L = end - start`, return the
        features' one-hot encoding corresponding to that region. For
        instance, for `n_features`, each position in that sequence will
        have a binary vector specifying whether the genomic feature's
        coordinates overlap with that position.
        @TODO: Clarify with an example, as this is hard to read right now.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        start : int
            The 0-based first position in the region.
        end : int
            One past the 0-based last position in the region.

        Returns
        -------
        numpy.ndarray
            :math:`L \\times N` array, where :math:`L = end - start`
            and :math:`N =` `self.n_features`. Note that if we catch a
            `tabix.TabixError`, we assume the error was the result of
            there being no features present in the queried region and
            return a `numpy.ndarray` of zeros.

        """

        return np.array([self._feature_handlers[i].stats(chrom, start, end) for i in self.features])
