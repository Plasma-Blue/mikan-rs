//! A **m**edical **i**mage **k**it for segment**a**tion metrics evaluatio**n**, native Rust support, and Python bindings for cross-language performance.
//!
//! ## 🎨 Features
//!
//! - 🚀 **Blazing Fast Speed**: Written in Rust with high parallelization; speeds are 1-50 times faster than current tools, especially for Hausdorff distance calculations.
//!
//! - 🧮 **Comprehensive Metrics**: Easily compute a wide range of segmentation metrics:
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
//! - 🐍 **Python Interface**: Provides Python bindings for seamless integration and cross-language performance.
//!
//! - 🎯 **Simple**: The API is so intuitive that you can use it right away without reading documentation! Whether using library functions, binary files, or Python functions, you have maximum flexibility in usage!
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
//! For details, please refer to the [rust examples](examples/tutorial.rs) and [python examples](examples/tutorial.py)。
//!
//! ## 🍚 Q&A
//!
//! Q: Why are my results different from medpy/seg_metrics/miseval/Metrics Reloaded?
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

pub use api::Evaluator;
