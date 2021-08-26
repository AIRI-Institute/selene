import pyBigWig
import numpy as np
from .target import Target


class qSeqGenomicFeatures(Target):
    """
    Stores the dataset specifying sequence regions and features.
    Accepts a path to directory containing bigWig files with feature values

    Parameters
    ----------
    features : list(str)
        non-redundant feature names
    features_path : list(str)
        locations of corresponding bigWig files
    target_length : int
        target length of returned feature values

    Attributes
    ----------
    features : list(str)
        non-redundant feature names.
    n_features : int
        The number of distinct features.
    target_length : int
        The number of bins to aggregate feature values over.
    """

    def __init__(self, features, features_path, target_length):
        """
        Constructs a new `qSeqGenomicFeatures` object.
        """

        self.features = features
        with open(features_path) as f:
            feature_path = dict(map(lambda x: x.strip().split("\t"), f.readlines()))
        feature_path = [feature_path[feature] for feature in self.features]

        self._feature_handlers = {
            i: pyBigWig.open(j) for i, j in zip(features, feature_path)
        }
        self.n_features = len(features)
        self.target_length = target_length

    def get_feature_data(self, chrom, start, end):
        """
        For a sequence of length :math:`L = end - start`, return the
        features' values corresponding to that region. Feature values
        are means of quantitative feature values binned into `self.target_length` bins.

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
            array of shape (T, N), where T is the target length,
            N is a number of features, and array[i][j] is the mean j-th feature signal
            in the i-th bin of interval (chrom, start, end).

        """
        return np.array(
            [
                self._feature_handlers[i].stats(
                    chrom, start, end, nBins=self.target_length
                )
                for i in self.features
            ]
        ).T
