from typing import Dict, List, Union

import numpy as np
import SimpleITK as sitk

from mikan._mikan import all_rs as _all
from mikan._mikan import calc_metrics_use_ndarray_rs as _metrics
from mikan._mikan import unique_rs as _unique
from mikan.alias import ALIAS_DICT


def all(gt: str, pred: str) -> List[dict]:
    '''
    Get All Metrics for All labels.
    '''
    return _all(gt, pred)
    

class LabelSelector:
    def __init__(self, evaluator: 'ArrayEvaluator', labels: Union[int, List[int]]):

        self.evaluator = evaluator
        self.labels = [labels] if isinstance(labels, int) else labels
    
    def metrics(self, metrics_names: Union[str, List[str]]) -> Union[float, List[float], Dict[str, Dict[str, float]]]:

        metrics_list = [metrics_names] if isinstance(metrics_names, str) else metrics_names
        
        # map alias
        required_base_metrics = {
            ALIAS_DICT[metric] 
            for metric in metrics_list
        }
        
        need_distance = any(
            dist_met in required_base_metrics for dist_met in ("hausdorff_distance", "hausdorff_distance_95", "assd", "masd")
        )
        results = self.evaluator._get_results(self.labels, need_distance)
        
        # re-map alias
        mapped_results = []
        for result in results:
            mapped_result = {}
            for metric in metrics_list:
                base_metric = ALIAS_DICT[metric]
                mapped_result[metric] = result[base_metric]
            mapped_results.append(mapped_result)
        
        # 1-label 1-metric -> float
        if isinstance(metrics_names, str) and len(self.labels) == 1:
            return mapped_results[0][metrics_names]
            
        # 1-label multi-metrics -> list[float]
        if isinstance(metrics_names, list) and len(self.labels) == 1:
            return [mapped_results[0][metric] for metric in metrics_names]
            
        # multi-labels 1-metric -> list[float]
        if isinstance(metrics_names, str):
            return [result[metrics_names] for result in mapped_results]
            
        # multi-labels multi-metrics -> dict
        return {
            str(label): {
                metric: result[metric]
                for metric in metrics_list
            }
            for label, result in zip(self.labels, mapped_results)
        }

class ArrayEvaluator:

    def __init__(self, gt_arr: np.ndarray, pred_arr: np.ndarray, spacing):

        self.gt_arr = gt_arr
        self.pred_arr = pred_arr
        self.spacing = spacing
        self._cache: Dict[int, Dict[str, float]] = {}
        
    def labels(self, labels: Union[int, List[int], str]) -> LabelSelector:
        
        if isinstance(labels, str):
            assert labels == "all"
            labels = set(_unique(self.gt_arr) + _unique(self.pred_arr))
            labels.discard(0)
        
        return LabelSelector(self, labels)
    
    def _get_results(self, labels: List[int], need_distance: bool = False) -> List[Dict[str, float]]:
        
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

    def __init__(self, gt: sitk.Image, pred: sitk.Image):

        assert gt.GetSpacing() == pred.GetSpacing(), "Spacing mismatch"
        assert gt.GetDirection() == pred.GetDirection(), "Direction mismatch"
        assert gt.GetSize() == pred.GetSize(), "Spacing mismatch"

        self.gt_arr = sitk.GetArrayFromImage(gt)
        self.pred_arr = sitk.GetArrayFromImage(pred)
        self.spacing = gt.GetSpacing()

        self._cache: Dict[int, Dict[str, float]] = {}

