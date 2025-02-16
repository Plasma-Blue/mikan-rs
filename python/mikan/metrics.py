import time

import numpy as np
from typing import List
from mikan._mikan import metrics_all_bind as _all
from rich import print
from medpy.metric import dc, hd

def all(gt: str, pred: str) -> List[dict]:
    return _all(gt, pred)


if __name__ == "__main__":

    # t = time.time()
    # ev = all(rf"data\patients_26_ground_truth.nii.gz", rf"data\patients_26_segmentation.nii.gz")
    # print(ev)
    # print(time.time()-t)

    import SimpleITK as sitk
    im = sitk.GetArrayFromImage(sitk.ReadImage(rf"data\patients_26_ground_truth.nii.gz"))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(rf"data\patients_26_segmentation.nii.gz"))
    t = time.time()
    for i in (1,2,3,4,5):
        pred_arr = (pred == i)
        gt_arr = (im == i)
        print(dc(pred_arr, gt_arr))
        print(hd(pred_arr, gt_arr))
        
    print(time.time() - t)
