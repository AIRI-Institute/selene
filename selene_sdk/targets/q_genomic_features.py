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
        non-redundant track names
    feature_paths_file : str
        tab-delimited file which allows to find correspondence between
        feature(=track) name and path to bigWig file with data for
        this track 
    agg_function : str
        aggregation function used for quantitative features, defines how
        to aggregate feature values across target genomic interval
        should be one of pyBigWig-supported aggregation functions

    Attributes
    ----------
    tracks : list(str)
        non-redundant feature names.
    n_features : int
        The number of distinct features.
    agg_function : str
        Aggregation function used for quantitative features
    """

    def __init__(self, features, features_path, agg_function="max"):
        """
        Constructs a new `qGenomicFeatures` object.
        """

        self.tracks =  features
        self.agg_function = agg_function
        self._feature_handlers = {}
        # features_path = dict(
        #         [line.strip().split("\t") \
        #             for line in open(feature_paths_file)
        #             ]
        #     )
        # features_path = [features_path[feature] \
        #                     for feature in self.tracks]

        for i,j in zip(features,features_path):
            try:
                self._feature_handlers[i] = pyBigWig.open(j)
            except Exception:
                print(traceback.format_exc())
                print ("Error dataset ",i," (#",len(self._feature_handlers),") from file ",j)
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
                                                                                    for i in self.tracks])
            return results
        except Exception:
            print(traceback.format_exc())
            print ("Error loading data on position ",chrom,start,end)
            sys.exit()