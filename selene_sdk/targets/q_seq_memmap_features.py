import os

import numpy as np

from .target import Target


class qSeqMemMapFeatures(Target):
    """
    Stores the dataset specifying sequence regions and features.
    Accepts a path to directory containing bigWig files with feature values

    Parameters
    ----------
    target_folder_path : str
        path to a dataset of chromosome-wise sample arrays
    memmap_shapes_path : str
        path to a file with array shapes for each chromosome file
    memmap_tracks_file : str
        path to file with a list of names of tracks stored in memmap arrays
    features : list(str)
        names of unique tracks to be retrieved

    Attributes
    ----------
    memmap_shapes : dict(str: tuple(int))
        shapes of each of the chromosome sample arrays
    memmaps : dict(str: np.memmap)
        readable `np.memmap` arrays for each chromosome
    tracks : list(str)
        names of unique tracks to be retrieved
    track_idxs : np.array
        indices of tracks to be retrieved in the `np.memmap` arrays
    """

    def __init__(
        self, 
        target_folder_path, 
        memmap_shapes_path,
        memmap_tracks_file,
        features
    ):
        """
        Constructs a new `qSeqMemMapFeatures` object.
        """

        memmap_shapes = {}
        with open(memmap_shapes_path) as f:
            for line in f:
                chrom, chrom_memmap_shape = line.rstrip().split('\t')
                memmap_shapes[chrom] = eval(chrom_memmap_shape)
        self.memmap_shapes = memmap_shapes

        self.memmaps = {}
        for target_path in os.listdir(target_folder_path):
            if not target_path.endswith('.arr'):
                continue
            chrom = target_path.split('.')[0]
            target_path = os.path.join(target_folder_path, target_path)

            self.memmaps[chrom] = np.memmap(
                target_path, 
                dtype='float32', 
                mode='r+', 
                shape=self.memmap_shapes[chrom]
            )

        """
        tracks = []
        with open(target_tracks_file) as f:
            for line in f:
                tracks.append(line.rstrip())
        """
        self.tracks = features


        with open(memmap_tracks_file) as f:
            self.memmap_tracks = list(map(lambda x: x.rstrip(), f.readlines()))
        
        track_idxs = []        
        for track in self.tracks:
            track_idxs.append(self.memmap_tracks.index(track))

        self.track_idxs = np.array(track_idxs)


    def get_feature_data(self, chrom, sample_idx):
        """
        For a sequence of length :math:`L = end - start`, return the
        features' values corresponding to that region. It is expected that 
        values provided by `self.memmaps` are means of binned quantitative 
        feature values for a set of samples.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        sample_idx : int
            Index of sample to retrieve in a given chromosome

        Returns
        -------
        numpy.ndarray
            array of shape (T, N), where T is the target length,
            N is a number of features, and array[i][j] is the mean j-th track signal
            in the i-th bin of interval corresponding to sample `sample_idx`.

        """
        return self.memmaps[chrom][self.track_idxs][sample_idx].T
