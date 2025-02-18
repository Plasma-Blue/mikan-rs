# mikan-rs

A **m**edical **i**mage **k**it for segment**a**tion metrics evaluatio**n**, native Rust support, and Python bindings for cross-language performance.

## 🎨 Features

- 🚀 **Blazing Fast**: Written in Rust with high parallelization; speeds are 10-200x faster than medpy (depends on the number of cores in your CPU), especially for Hausdorff distance calculations.

- 🎯 **Simple**: The API is so intuitive that you can start using it immediately while reading the [documentation](examples/tutorial.ipynb) in just one minute!

- 🧮 **Comprehensive Metrics**: Easily to compute almost all of segmentation metrics:

  - **Confusion Matrix Based:**

    - Dice/IoU
    - TP/TN/FP/FN
    - Sensitivity/Specificity/Precision
    - Accuracy/Balanced Accuracy
    - ARI/FNR/FPR/F-score
    - Volume Similarity
    - MCC/nMCC/aMCC

  - **Distance Based:**
    - Hausdorff Distance (HD)
    - Hausdorff Distance 95 (HD95)
    - Average Symmetric Surface Distance (ASSD)
    - Mean Average Surface Distance (MASD)

## 🔨 Install

`cargo add mikan-rs` for rust project.

`pip install mikan-rs` for python.

## 🥒 Develop

`maturin dev`

## 📘 Usages

```python
import mikan
import SimpleITK as sitk

gt = sitk.ReadImage("gt.nii.gz", sitk.sitkUInt8)
pred = sitk.ReadImage("pred.nii.gz", sitk.sitkUInt8)

e = mikan.Evaluator(gt, pred)
e.labels(1).metrics("dice")
```

For details, please refer to the [python examples](examples/tutorial.ipynb) and [rust examples](examples/tutorial.rs).

## 🍚 Q&A

Q: Why are my results different from seg_metrics/miseval/Metrics Reloaded?

A: They are wrong. Of course, we might be wrong too. PRs to fix issues are welcome!

## 🔒 License

Licensed under either of the following licenses, at your choice:

Apache License, Version 2.0
(See LICENSE-APACHE or visit http://www.apache.org/licenses/LICENSE-2.0)

MIT License
(See LICENSE-MIT or visit http://opensource.org/licenses/MIT)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project, as defined by the Apache License 2.0, will be dual-licensed under the above licenses without any additional terms or conditions.
