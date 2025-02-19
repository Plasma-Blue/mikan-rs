//! A **m**edical **i**mage **k**it for segment**a**tion metrics evaluatio**n**, native Rust support, and Python bindings for cross-language performance.
//!
//! ## 🎨 Features
//!
//! - 🚀 **Blazingly Fast**: Written in Rust with high parallelization; speeds are 10-200x faster than medpy (depends on the number of cores in your CPU), especially for Hausdorff distance calculations.
//!
//! - 🎯 **Simple**: The API is so intuitive that you can start using it immediately while reading the [documentation](https://github.com/Plasma-Blue/mikan-rs/blob/master/examples/tutorial.ipynb) in just one minute!
//!
//! - 🧮 **Comprehensive Metrics**: Easily to compute almost all segmentation metrics:
//!
//!   - **Confusion Matrix Based:**
//!
//!     - Dice/IoU
//!     - TP/TN/FP/FN
//!     - Sensitivity/Specificity/Precision
//!     - Accuracy/Balanced Accuracy
//!     - ARI/FNR/FPR/F-score
//!     - Volume Similarity
//!     - MCC/nMCC/aMCC
//!
//!   - **Distance Based:**
//!     - Hausdorff Distance (HD)
//!     - Hausdorff Distance 95 (HD95)
//!     - Average Symmetric Surface Distance (ASSD)
//!     - Mean Average Surface Distance (MASD)
//!
//! ## 🔨 Install
//!
//! `cargo add mikan-rs` for rust project.
//!
//! `pip install mikan-rs` for python.
//!
//! ## 🥒 Develop
//!
//! `maturin dev`
//!
//! ## 📘 Usages
//!
//! For details, please refer to the [rust examples](https://github.com/Plasma-Blue/mikan-rs/blob/master/examples/tutorial.rs) and [python examples](https://github.com/Plasma-Blue/mikan-rs/blob/master/examples/tutorial.ipynb)。
//!
//! ## 🍚 Q&A
//!
//! Q: Why are my results different from seg_metrics/miseval/Metrics Reloaded?
//!
//! A: They are wrong. Of course, we might be wrong too. PRs to fix issues are welcome!
//!
//! ## 🔒 License
//!
//! Apache-2.0 & MIT

pub mod api;
mod bind;
mod metrics;
mod utils;

pub use api::{all, metrics, Evaluator};
