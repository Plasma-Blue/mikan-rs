"""
Module for evaluating segmentation metrics.

This module provides classes and functions to evaluate segmentation metrics
using ground truth and predicted images.

Functions
---------
all(gt: str, pred: str) -> List[dict]
    Get all metrics for all labels.

Classes
-------
LabelSelector
    A class to select labels and compute metrics for them.
ArrayEvaluator
    A class to evaluate metrics using numpy arrays.
Evaluator
    A class to evaluate metrics using SimpleITK images.

Examples
--------
>>> import SimpleITK as sitk
>>> from mikan.evaluator import Evaluator

>>> # Load ground truth and predicted images
>>> gt_image = sitk.ReadImage('gt.nii.gz', sitk.sitkUInt8)
>>> pred_image = sitk.ReadImage('pred.nii.gz', sitk.sitkUInt8)

>>> # Initialize the Evaluator
>>> evaluator = Evaluator(gt_image, pred_image)

>>> # Select a single label and compute a single metric
>>> dice_score = evaluator.labels(1).metrics('dice')
>>> print(dice_score)
0.85

>>> # Select multiple labels and compute a single metric for each
>>> dice_scores = evaluator.labels([1, 2, 3])。metrics('dice')
>>> print(dice_scores)
[0.85, 0.78, 0.80]

>>> # Select a single label and compute multiple metrics
>>> metrics = evaluator.labels(1).metrics(['dice', 'iou'])
>>> print(metrics)
[0.85, 0.75]

>>> # Select multiple labels and compute multiple metrics for each
>>> label_selector = evaluator.labels([1, 2]).metrics(['dice', 'iou'])
>>> print(metrics)
{'1': {'dice': 0.85, 'iou': 0.75}, '2': {'dice': 0.78, 'iou': 0.68}}

>>> # Select all labels and compute all metrics
>>> label_selector = evaluator.labels('all').metrics('all')
>>> print(all_metrics)
{
    '1': {'dice': 0.85, 'iou': 0.75, 'hausdorff_distance': 2.0, ...},
    '2': {'dice': 0.78, 'iou': 0.68, 'hausdorff_distance': 3.5, ...},
    ...
}
"""

from typing import Dict, List, Union

import numpy as np
import SimpleITK as sitk

from mikan._mikan import all_rs as _all
from mikan._mikan import calc_metrics_use_ndarray_rs as _metrics
from mikan._mikan import unique_rs as _unique
from mikan.alias import ALIAS_DICT

__all__ = ["all", "Evaluator", "ArrayEvaluator"]


def all(gt: str, pred: str) -> List[dict]:
    """
    Get all metrics for all labels.

    Parameters
    ----------
    gt : str
        Path to the ground truth image.
    pred : str
        Path to the predicted image.

    Returns
    -------
    List[dict]
        A list of dictionaries containing metrics for each label.
    """
    return _all(gt, pred)


class LabelSelector:
    """
    A class to select labels and compute metrics for them.

    Parameters
    ----------
    evaluator : ArrayEvaluator
        An instance of ArrayEvaluator.
    labels : Union[int, List[int]]
        A single label or a list of labels to evaluate.

    Methods
    -------
    metrics(metrics_names: Union[str, List[str]]) -> Union[float, List[float], Dict[str, Dict[str, float]]]
        Compute metrics for the selected labels.
    """
    
    def __init__(self, evaluator: 'ArrayEvaluator', labels: Union[int, List[int]]):
        self.evaluator = evaluator
        self.labels = [labels] if isinstance(labels, int) else labels
    
    def metrics(self, metrics_names: Union[str, List[str]]) -> Union[float, List[float], Dict[str, Dict[str, float]]]:
        """
        Compute metrics for the selected labels.

        Parameters
        ----------
        metrics_names : Union[str, List[str]]
            A single metric name or a list of metric names to compute.

        Returns
        -------
        Union[float, List[float], Dict[str, Dict[str, float]]]
            Computed metrics. The return type depends on the number of labels and metrics.
        """
        if isinstance(metrics_names, str):
            if metrics_names == "all":
                metrics_list = list(set(ALIAS_DICT.values()))
            else:
                metrics_list = [metrics_names] 
        else:
            metrics_list = metrics_names
        
        try:
            required_base_metrics = {
                ALIAS_DICT[metric] 
                for metric in metrics_list
            }
        except KeyError as e:
            raise KeyError(f"{e} not in metric dicts. Check it !")
        
        need_distance = any(
            dist_met in required_base_metrics for dist_met in ("hausdorff_distance", "hausdorff_distance_95", "assd", "masd")
        )
        results = self.evaluator._get_results(self.labels, need_distance)
        
        mapped_results = []
        for result in results:
            mapped_result = {}
            for metric in metrics_list:
                base_metric = ALIAS_DICT[metric]
                mapped_result[metric] = result[base_metric]
            mapped_results.append(mapped_result)
        
        if isinstance(metrics_names, str) and len(self.labels) == 1:
            return mapped_results[0][metrics_names]
            
        if isinstance(metrics_names, list) and len(self.labels) == 1:
            return [mapped_results[0][metric] for metric in metrics_names]
            
        if isinstance(metrics_names, str) and metrics_names != "all":
            return [result[metrics_names] for result in mapped_results]
            
        return {
            str(label): {
                metric: result[metric]
                for metric in metrics_list
            }
            for label, result in zip(self.labels, mapped_results)
        }


class ArrayEvaluator:
    """
    A class to evaluate metrics using numpy arrays.

    Parameters
    ----------
    gt_arr : np.ndarray
        Ground truth image as a numpy array.
    pred_arr : np.ndarray
        Predicted image as a numpy array.
    spacing : tuple
        Spacing of the images.

    Methods
    -------
    labels(labels: Union[int, List[int], str]) -> LabelSelector
        Select labels to evaluate.
    _get_results(labels: List[int], need_distance: bool = False) -> List[Dict[str, float]]
        Get evaluation results for the selected labels.
    """
    
    def __init__(self, gt_arr: np.ndarray, pred_arr: np.ndarray, spacing):
        assert gt_arr.shape == pred_arr.shape, "Array shape mismatch"
        self.gt_arr = gt_arr
        self.pred_arr = pred_arr
        self.spacing = spacing
        self._cache: Dict[int, Dict[str, float]] = {}
        
    def labels(self, labels: Union[int, List[int], str]) -> LabelSelector:
        """
        Select labels to evaluate.

        Parameters
        ----------
        labels : Union[int, List[int], str]
            A single label, a list of labels, or 'all' to select all labels.

        Returns
        -------
        LabelSelector
            An instance of LabelSelector for the selected labels.
        """
        if isinstance(labels, str):
            assert labels == "all"
            labels = set(_unique(self.gt_arr) + _unique(self.pred_arr))
            labels.discard(0)
        
        return LabelSelector(self, labels)
    
    def _get_results(self, labels: List[int], need_distance: bool = False) -> List[Dict[str, float]]:
        """
        Get evaluation results for the selected labels.

        Parameters
        ----------
        labels : List[int]
            A list of labels to evaluate.
        need_distance : bool, optional
            Whether distance metrics are needed, by default False.

        Returns
        -------
        List[Dict[str, float]]
            A list of dictionaries containing metrics for each label.
        """
        uncached_labels = []
        for label in labels:
            if label not in self._cache:
                uncached_labels.append(label)
            elif need_distance and 'hausdorff_distance' not in self._cache[label]:
                uncached_labels.append(label)
        
        if uncached_labels:
            new_results = _metrics(
                self.gt_arr, 
                self.pred_arr, 
                uncached_labels, 
                self.spacing,
                need_distance
            )
            
            for result in new_results:
                label = int(result['label'])
                if label not in self._cache:
                    self._cache[label] = {}
                self._cache[label].update(result)
        
        return [self._cache[label] for label in labels]


class Evaluator(ArrayEvaluator):
    """
    A class to evaluate metrics using SimpleITK images.

    Parameters
    ----------
    gt : sitk.Image
        Ground truth image.
    pred : sitk.Image
        Predicted image.

    Methods
    -------
    Inherits all methods from ArrayEvaluator.
    """
    
    def __init__(self, gt: sitk.Image, pred: sitk.Image):
        assert gt.GetSpacing() == pred.GetSpacing(), "Spacing mismatch"
        assert gt.GetDirection() == pred.GetDirection(), "Direction mismatch"
        assert gt.GetSize() == pred.GetSize(), "Size mismatch"

        self.gt_arr = sitk.GetArrayFromImage(gt)
        self.pred_arr = sitk.GetArrayFromImage(pred)
        self.spacing = gt.GetSpacing()

        self._cache: Dict[int, Dict[str, float]] = {}
