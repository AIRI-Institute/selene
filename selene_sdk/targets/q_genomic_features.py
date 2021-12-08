import pyBigWig
import traceback
import sys
import numpy as np
from .target import Target

class qGenomicFeatures(Target):
    """
    Stores the dataset specifying sequence regions and features.
    Accepts a path to directory containing bigWig files with feature values

    Parameters
    ----------
    features : list(str)
        non-redundant feature names
    features_path : list(str)
        locations of corresponding bigWig files
    agg_function : str
        aggregation function used for quantitative features, defines how
        to aggregate feature values across target genomic interval
        should be one of pyBigWig-supported aggregation functions

    Attributes
    ----------
    features : list(str)
        non-redundant feature names.
    n_features : int
        The number of distinct features.
    agg_function : str
        Aggregation function used for quantitative features
    """

    def __init__(self, features, features_path,agg_function):
        """
        Constructs a new `qGenomicFeatures` object.
        """

        self.features =  features
        self.agg_function = agg_function
        self._feature_handlers = {}
        for i,j in zip(features,features_path):
            try:
                self._feature_handlers[i] = pyBigWig.open(j)
            except Exception:
                print(traceback.format_exc())
                print ("Error dataset ",i," from file ",j)
                sys.exit()


    def get_feature_data(self, chrom, start, end):
        """
        For a sequence of length :math:`L = end - start`, return the
        features' values corresponding to that region. Feature values
        are maximum of quantitative feature values computed observed in
        the specified interval.

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
            array of length N, where N is a number of features, and
            array[i] is a max of feature signal over the input genomic
            interval.

        """

        try:
            results = np.array([self._feature_handlers[i].stats(chrom, start, end, type=self.agg_function)[0] \
                                                                                    for i in self.features])
            return results
        except Exception:
            print(traceback.format_exc())
            print ("Error loading data on position ",chrom,start,end)
            sys.exit()