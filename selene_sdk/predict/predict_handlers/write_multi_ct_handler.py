"""
Handles outputting the model predictions
"""
import pyBigWig
import os
from tqdm import tqdm
from .handler import PredictionsHandler

class WritePredictionsMultiCtBigWigHandler(PredictionsHandler):
    """
    Collects batches of model predictions and writes all of them
    to file at the end.

    Parameters
    ----------
    features : list(str)
        List of sequence-level features, in the same order that the
        model will return its predictions.
    big_wig_column_ids : list(str)
        list of ids containing info requered for bigWig: [id_chrm_name, id_start, id_end]
    cell_type_names : list(str)
        names of cell types, will be used as file suffix
        note that cell types should appear in prediction results
        in the same order as they are listed here
    bw_header : bigWig header, list(("chr",size))
        i.e. [("chr1", 1000000), ("chr2", 1500000)]
    output_path_prefix : str
        Path to the file to which Selene will write the absolute difference
        scores. The path may contain a filename prefix. Selene will append
        `predictions` to the end of the prefix.
    output_format : {'bigWig'}
        Specify the desired output format. Currently only bigWig supported.
    write_mem_limit : int, optional
        Default is 1500. Specify the amount of memory you can allocate to
        storing model predictions/scores for this particular handler, in MB.
        Handler will write to file whenever this memory limit is reached.

    """

    def __init__(self,
                 features,
                 big_wig_column_ids,
                 cell_type_column_id,
                 cell_type_names,
                 bw_header,
                 output_path_prefix,
                 write_mem_limit=1500):
        """
        Constructs a new `WritePredictionsHandler` object.
        """

        self._cell_type_names = cell_type_names
        self._features = features
        self._cell_type_column_id = cell_type_column_id
        self._big_wig_column_ids = big_wig_column_ids
        self._handlers = {}

        # create bigWig file handlers
        for ct in self._cell_type_names:
            self._handlers[ct] = {}
            for feature in self._features:
                fname = ct+"_"+feature+".bw"
                self._handlers[ct][feature] = pyBigWig.open(output_path_prefix+fname,
                                                            "w")
                self._handlers[ct][feature].addHeader(bw_header)
        self._results = []
        self._samples = []

        self._write_mem_limit = write_mem_limit

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids):
        """
        Handles the predictions for a batch of sequences.

        Parameters
        ----------
        batch_predictions : arraylike
            The predictions for a batch of sequences. This should have
            dimensions of :math:`B \\times N` (where :math:`B` is the
            size of the mini-batch and :math:`N` is the number of
            features).
        batch_ids : list(arraylike)
            Batch of sequence identifiers. Each element is `arraylike`
            because it may contain more than one column (written to
            file) that together make up a unique identifier for a
            sequence.
        """
        self._results.append(batch_predictions)
        self._samples.append(batch_ids)
        if self._reached_mem_limit():
            self.write_to_file()

    def write_to_file(self):
        """
        Writes the stored scores to a file.

        """

        # TODO really not very effective, just a simplest solution for now
        for batch_ids,batch_predictions in zip(tqdm(self._samples), self._results):
            for metadata,targets in zip(batch_ids,batch_predictions):
                ct = self._cell_type_names[metadata[self._cell_type_column_id]]
                chrm, start, end = [metadata[i] for i in self._big_wig_column_ids]
                for feature_id,feature in enumerate(self._features):
                    self._handlers[ct][feature].addEntries(str(chrm),
                                                           [int(start+end)//2],
                                                           #ends=[(int(start)+int(end))//2+1],
                                                           values=[float(targets[feature_id])],
                                                           span=1
                                                           )

    def close_handlers(self):
        """
        Close opened handlers
        For files opened for writing, closing a file writes any buffered entries to disk,
        constructs and writes the file index, and constructs zoom levels. Consequently, this
        can take a bit of time.

        """

        for i in self._handlers.values():
            for j in i.values():
                j.close()


