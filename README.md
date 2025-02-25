# mikan-rs 🍊

A **m**edical **i**mage **k**it for segment**a**tion metrics evaluatio**n**, native Rust support, and Python bindings for cross-language performance.

## 🎨 Features

- 🚀 **Blazingly Fast**: Written in Rust with high parallelization; speeds are 10-100x faster than medpy (depends on the number of cores in your CPU), especially for Hausdorff distance calculations.

- 🎯 **Simple**: The API is so intuitive that you can start using it immediately while reading the [documentation](https://github.com/Plasma-Blue/mikan-rs/blob/master/examples/tutorial.ipynb) in just one minute!

- 🧮 **Comprehensive Metrics**: Easily to compute almost all of segmentation metrics, **results are consistent with medpy**:

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

For Rust projects, add the following to your `Cargo.toml`:

```toml
[dependencies]
mikan-rs = "*"
```

For Python, install via pip:

```sh
pip install mikan-rs
```

## 🥒 Develop

`maturin dev`

## 📘 Usages

```python
import mikan
import SimpleITK as sitk

gt = sitk.ReadImage("gt.nii.gz", sitk.sitkUInt8)
pred = sitk.ReadImage("pred.nii.gz", sitk.sitkUInt8)

e = mikan.Evaluator(gt, pred)
e.labels([1, 2, 3]).metrics(["dice", "hd", "hd95", "assd"])
```

For details, please refer to the [python examples](https://github.com/Plasma-Blue/mikan-rs/blob/master/examples/tutorial.ipynb) and [rust examples](https://github.com/Plasma-Blue/mikan-rs/blob/master/examples/tutorial.rs).

## 🎄 Related Projects

- [medpy](https://github.com/loli/medpy): A well-known package for calculating segmentation metrics, with excellent documentation and implementation.
- [miseval](https://github.com/frankkramer-lab/miseval): A framework capable of calculating a large number of segmentation metrics.
- [seg_metrics](https://github.com/Jingnan-Jia/segmentation_metrics): A package for segmentation metrics that supports batch data calculation and CSV output, making it very convenient.
- [MetricsReloaded](https://github.com/Project-MONAI/MetricsReloaded): A new recommendation framework for biomedical image analysis validation, published in Nature Methods.

## 📃 Citation

If you use this software, we would appreciate it if you could include an mikan emoji 🍊 in your paper.

## 🍚 Q&A

Q: Why are my results different from seg_metrics/miseval/MetricsReloaded?

A: They are wrong. Of course, we might be wrong too. PRs to fix issues are welcome!

## 🔒 License

Licensed under either of the following licenses, at your choice:

Apache License, Version 2.0
(See LICENSE-APACHE or visit <http://www.apache.org/licenses/LICENSE-2.0>)

MIT License
(See LICENSE-MIT or visit <http://opensource.org/licenses/MIT>)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project, as defined by the Apache License 2.0, will be dual-licensed under the above licenses without any additional terms or conditions.
