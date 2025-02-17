import mikan
import SimpleITK as sitk
import time
from medpy.metric import dc, hd, hd95, assd

gt = sitk.ReadImage(rf"data\patients_26_ground_truth.nii.gz", sitk.sitkUInt8)
pred = sitk.ReadImage(rf"data\patients_26_segmentation.nii.gz", sitk.sitkUInt8)

gt_arr = sitk.GetArrayFromImage(gt)
pred_arr = sitk.GetArrayFromImage(pred)

# mikan: DSC
t = time.time()
evaluator = mikan.ArrayEvaluator(gt_arr, pred_arr, spacing=gt.GetSpacing())
dsc = evaluator.labels([1,2,3,4,5]).metrics("dsc")
mikan_costs = time.time() - t

# medpy: DSC
t = time.time()
for i in (1,2,3,4,5):
    dsc = dc(pred_arr == i, gt_arr == i)
medpy_costs = time.time() - t

print(f"DSC: {medpy_costs / mikan_costs :.2f}x faster")


# mikan: HD
t = time.time()
evaluator = mikan.ArrayEvaluator(gt_arr, pred_arr, spacing=gt.GetSpacing())
hausdorff_distance = evaluator.labels([1,2,3,4,5]).metrics("HD")
mikan_costs = time.time() - t
print(f"Mikan has calculated Hausdorff distance and cost {mikan_costs:.2f} s.")
print(f"Let's waiting for medpy, be patient for a while...")

# medpy: HD
t = time.time()
for i in (1,2,3,4,5):
    hausdorff_distance = hd(pred_arr == i, gt_arr == i, voxelspacing=gt.GetSpacing())
medpy_costs = time.time() - t

print(f"HD: {medpy_costs / mikan_costs :.2f}x faster")
