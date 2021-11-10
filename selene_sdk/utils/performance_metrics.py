"""
This module provides the `PerformanceMetrics` class and supporting
functionality for tracking and computing model performance.
"""
from collections import defaultdict, namedtuple
import logging
import os
import warnings
import pandas as pd

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import rankdata


logger = logging.getLogger("selene")


Metric = namedtuple("Metric", ["fn", "transform", "data"])
"""
A tuple containing a metric function and the results from applying that
metric to some values.

Parameters
----------
fn : types.FunctionType
    A metric.
transfrom : types.FunctionType
    A transform function which should be applied to data before measuring metric
data : list(float)
    A list holding the results from applying the metric.

Attributes
----------
fn : types.FunctionType
    A metric.
transfrom : types.FunctionType
    A transform function which should be applied to data before measuring metric
data : list(float)
    A list holding the results from applying the metric.

"""


def visualize_roc_curves(prediction,
                         target,
                         output_dir,
                         target_mask=None,
                         report_gt_feature_n_positives=50,
                         style="seaborn-colorblind",
                         fig_title="Feature ROC curves",
                         dpi=500):
    """
    Output the ROC curves for each feature predicted by a model
    as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    report_gt_feature_n_positives : int, optional
        Default is 50. Do not visualize an ROC curve for a feature with
        less than 50 positive examples in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature ROC curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.figure()

    n_features = prediction.shape[-1]
    for index in range(n_features):
        feature_preds = prediction[..., index]
        feature_targets = target[..., index]
        if target_mask is not None:
            feature_mask = target_mask[..., index]
            # if mask is n_samples x n_cell_types,
            # feature_targets and feature_preds get flattened but that's ok
            # b/c each item is a separate sample anyway
            feature_targets = feature_targets[feature_mask]
            feature_preds = feature_preds[feature_mask]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
            plt.plot(fpr, tpr, 'r-', color="black", alpha=0.3, lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if fig_title:
        plt.title(fig_title)
    plt.savefig(os.path.join(output_dir, "roc_curves.svg"),
                format="svg",
                dpi=dpi)


def visualize_precision_recall_curves(
        prediction,
        target,
        output_dir,
        target_mask=None,
        report_gt_feature_n_positives=50,
        style="seaborn-colorblind",
        fig_title="Feature precision-recall curves",
        dpi=500):
    """
    Output the precision-recall (PR) curves for each feature predicted by
    a model as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    report_gt_feature_n_positives : int, optional
        Default is 50. Do not visualize an PR curve for a feature with
        less than 50 positive examples in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature precision-recall curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """
    os.makedirs(output_dir, exist_ok=True)

    # TODO: fix this
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt

    plt.style.use(style)
    plt.figure()

    n_features = prediction.shape[-1]
    for index in range(n_features):
        feature_preds = prediction[..., index]
        feature_targets = target[..., index]
        if target_mask is not None:
            feature_mask = target_mask[..., index]
            # if mask is n_samples x n_cell_types,
            # feature_targets and feature_preds get flattened but that's ok
            # b/c each item is a separate sample anyway
            feature_targets = feature_targets[feature_mask]
            feature_preds = feature_preds[feature_mask]
        if len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            precision, recall, _ = precision_recall_curve(
                feature_targets, feature_preds)
            plt.step(
                recall, precision, 'r-',
                color="black", alpha=0.3, lw=1, where="post")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if fig_title:
        plt.title(fig_title)
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.svg"),
                format="svg",
                dpi=dpi)


def compute_score(prediction, target, metric_fn, target_mask=None,
                  report_gt_feature_n_positives=10):
    """
    Using a user-specified metric, computes the distance between
    two tensors.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    metric_fn : types.FunctionType
        A metric that can measure the distance between the prediction
        and target variables.
    target_mask: numpy.ndarray, optional
        A mask of shape `target.shape` that indicates which values
        should be considered when computing the scores.
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.

    Returns
    -------
    average_score, feature_scores : tuple(float, numpy.ndarray)
        A tuple containing the average of all feature scores, and a
        vector containing the scores for each feature. If there were
        no features meeting our filtering thresholds, will return
        `(None, [])`.
    """
    # prediction_shape:
    # batch_size*n_batches, n_cell_types, n_features
    n_features = prediction.shape[-1]
    n_cell_types = prediction.shape[1]

    track_scores = np.ones(shape=(n_cell_types,n_features)) * np.nan
  
    for feature_index in range(n_features):
        for cell_type_index in range(n_cell_types):
            feature_preds = np.ravel(prediction[:, cell_type_index, feature_index])
            feature_targets = np.ravel(target[:, cell_type_index, feature_index])
            if target_mask is not None:
                track_masks_arr = target_mask[:, cell_type_index, feature_index]
                
                # we assume that if track is masked, it is masked for all sequences
                track_mask = np.ravel(track_masks_arr)[0]
                assert np.all(track_masks_arr==track_mask)
                
                if not track_mask: # track was not measured or is masked
                    # should put nan into feature_scores: 
                    # feature_scores[cell_type_index,feature_index] = np.nan
                    # but it's already filled with nans so just continue
                    continue
            if len(np.unique(feature_targets)) > 0 and \
                np.count_nonzero(feature_targets) > report_gt_feature_n_positives:
                try:
                    track_scores[cell_type_index,feature_index] = metric_fn(
                        feature_targets, feature_preds)
                except ValueError:  # do I need to make this more generic?
                    continue
    
    # now we compute average score for all features
    # if all elements of feature_scores are nans
    # following will produce warning and return np.nan, 
    # which we just ignore
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        average_score = np.nanmean(track_scores)
    if np.isnan(average_score):
        return None, track_scores
    else:
        return average_score, track_scores

def get_feature_specific_scores(data, 
                                get_feature_from_index_fn,
                                get_ct_from_index_fn):
    """
    Generates a dictionary mapping feature names to feature scores from
    an intermediate representation.

    Parameters
    ----------
    data : list(tuple(int, float))
        A list of tuples, where each tuple contains a feature's index
        and the score for that feature.
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    get_ct_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a cell type
        name (`str`).

    Returns
    -------
    dict
        A dictionary mapping feature names (`str`) to scores (`float`).
        If there was no score for a feature, its score will be set to
        `None`.

    """
    feature_score_dict = {}
    for index, score in enumerate(data):
        feature = get_feature_from_index_fn(index)
        if not np.isnan(score):
            feature_score_dict[feature] = score
        else:
            feature_score_dict[feature] = None
    return feature_score_dict


def auc_u_test(labels, predictions):
    """
    Outputs the area under the the ROC curve associated with a certain 
    set of labels and the predictions given by the training model.
    Computed from the U statistic.

    Parameters
    ----------
    labels: numpy.ndarray
        Known labels of values predicted by model. Must be one dimensional.
    predictions: numpy.ndarray
        Value predicted by user model. Must be one dimensional, with matching
        dimension to `labels`

    Returns
    -------
    float
        AUC value of given label, prediction pairs  
   
    """
    len_pos = int(np.sum(labels))
    len_neg = len(labels) - len_pos
    rank_sum = np.sum(rankdata(predictions)[labels == 1])
    u_value = rank_sum - (len_pos * (len_pos + 1)) / 2
    auc = u_value / (len_pos * len_neg)
    return auc

class PerformanceMetrics(object):
    """
    Tracks and calculates metrics to evaluate how closely a model's
    predictions match the true values it was designed to predict.

    Parameters
    ----------
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    get_ct_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a cell type
        name (`str`).
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions.
        By default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, and `"average_precision"`,
        which maps to `sklearn.metrics.average_precision_score`.



    Attributes
    ----------
    skip_threshold : int
        The minimum number of positive examples of a feature that must
        be included in an update for a metric score to be
        calculated for it.
    get_feature_from_index : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    get_feature_from_index : types.FunctionType
        A function that takes an index (`int`) and returns a cell type
        name (`str`).
    metrics : dict
        A dictionary that maps metric names (`str`) to metric objects
        (`Metric`). By default, this contains `"roc_auc"` and
        `"average_precision"`.
    metrics_transforms: dict
        A dictionary mapping metrics name to transformation function,
        which should be applied tp to data prior to metrics computation.
    """

    def __init__(self,
                 get_feature_from_index_fn,
                 get_ct_from_index_fn,
                 report_gt_feature_n_positives=10,
                 metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score),
                 metrics_transforms=dict(roc_auc=None, 
                                         average_precision=None)):
        """
        Creates a new object of the `PerformanceMetrics` class.
        """
        self.skip_threshold = report_gt_feature_n_positives
        self.get_feature_from_index = get_feature_from_index_fn
        self.get_ct_from_index = get_ct_from_index_fn
        self.metrics = dict()
        for k, v in metrics.items():
            if k in metrics_transforms:
                self.metrics[k] = Metric(fn=v, 
                                         transform=metrics_transforms[k],
                                         data=[])
            else:
                self.metrics[k] = Metric(fn=v, 
                                         transform=None,
                                         data=[])

    def add_metric(self, name, metric_fn, transform_function = None):
        """
        Begins tracking of the specified metric.

        Parameters
        ----------
        name : str
            The name of the metric.
        metric_fn : types.FunctionType
            A metric function.
        transform_function: types.FunctionType
            A tranform function which should be
            applied to data before metrics computation
            if None, no transform will be applied
        """
        self.metrics[name] = Metric(fn=metric_fn, 
                                        transform=transform_function,
                                        data=[])

    def remove_metric(self, name):
        """
        Ends the tracking of the specified metric, and returns the
        previous scores associated with that metric.

        Parameters
        ----------
        name : str
            The name of the metric.

        Returns
        -------
        list(float)
            The list of feature-specific scores obtained by previous
            uses of the specified metric.

        """
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, prediction, target, target_mask=None):
        """
        Evaluates the tracked metrics on a model prediction and its
        target value, and adds this to the metric histories.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.
        target_mask : numpy.ndarray, optional
            A mask of shape `target.shape` that indicates which values
            should be considered when computing the scores.

        Returns
        -------
        dict
            A dictionary mapping each metric names (`str`) to the
            average score of that metric across all features
            (`float`).

        """
        metric_scores = {}
        for name, metric in self.metrics.items():
            if metric.transform is not None:
                tr_prediction, tr_target, tr_target_mask = metric.transform((prediction, target, target_mask))
            else:
                tr_prediction, tr_target, tr_target_mask = prediction, target, target_mask
            assert tr_prediction.shape == tr_target.shape == tr_target_mask.shape
            avg_score, track_scores = compute_score(
                tr_prediction, tr_target, metric.fn, target_mask=tr_target_mask,
                report_gt_feature_n_positives=self.skip_threshold)
            metric.data.append(track_scores)
            metric_scores[name] = avg_score
        return metric_scores

    def visualize(self, prediction, target, output_dir, target_mask=None, **kwargs):
        """
        Outputs ROC and PR curves. Does not support other metrics
        currently.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.
        output_dir : str
            The path to the directory to output the figures. Directories that
            do not currently exist will be automatically created.
        **kwargs : dict
            Keyword arguments to pass to each visualization function. Each
            function accepts the following args:

                * style : str - Default is "seaborn-colorblind". Specify a \
                          style available in \
                          `matplotlib.pyplot.style.available` to use.
                * dpi : int - Default is 500. Specify dots per inch \
                              (resolution) of the figure.

        Returns
        -------
        None
            Outputs figures to `output_dir`.

        """
        print ("This function is not consistent with new transform functions")
        raise NotImplementedError
        os.makedirs(output_dir, exist_ok=True)
        if "roc_auc" in self.metrics:
            visualize_roc_curves(
                prediction, target, output_dir, target_mask,
                report_gt_feature_n_positives=self.skip_threshold,
                **kwargs)
        if "average_precision" in self.metrics:
            visualize_precision_recall_curves(
                prediction, target, output_dir, target_mask,
                report_gt_feature_n_positives=self.skip_threshold,
                **kwargs)

    def write_feature_scores_to_file(self, output_path):
        """
        Writes each metric's score for each feature to a specified
        file.

        Parameters
        ----------
        output_path : str
            The path to the output file where performance metrics will
            be written.

        Returns
        -------
        pd.DataFeame
            A dataFrame with columns:
            cell_type,feature,metric_name,value

        """
        feature_scores = defaultdict(dict)
        full_metrics_results = []
        for name, metric in self.metrics.items():
            # metric.data contains n_cell_type x n_features array
            # of metric value computed for track[i,j]
            n_cell_types = metric.data[-1].shape[0]
            n_features = metric.data[-1].shape[1]
            cell_type_names = [self.get_ct_from_index(ct) for ct in range(n_cell_types)]
            feature_names = [self.get_feature_from_index(f) for f in range(n_features)]

            metric_df = pd.DataFrame(metric.data[-1],
                                        index=cell_type_names,
                                        columns=feature_names).reset_index()
            metric_df = pd.melt(metric_df, id_vars='index').rename(
                columns={"index":"cell_type","variable":"feature"}
            )
            metric_df["metric_name"] = [name]*len(metric_df)
            full_metrics_results.append(metric_df) 
        full_metrics_results = pd.concat(full_metrics_results)
        full_metrics_results[["metric_name","cell_type","feature","value"]].to_csv(output_path, 
                                        sep="\t", index=False)

        return full_metrics_results
