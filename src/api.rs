//! Core APIs. Carefully designed to be simple and easy to use.

use crate::metrics::{ConfusionMatrix, Distance};
use crate::utils::{get_unique_labels_parallel, merge_vector};
use std::collections::BTreeMap;

use nii::Nifti1Image;
use once_cell::unsync::OnceCell;
use rayon::prelude::*;

#[cfg_attr(doc, katexit::katexit)]
/// A struct to calculate various metrics based on confusion matrix & distance for segmentation tasks, designed to provide fine-grained computational control, such as determining target labels and target metrics.
pub struct Evaluator<'a> {
    cm: ConfusionMatrix,
    dist: OnceCell<Distance>,
    gt: &'a Nifti1Image<u8>,
    pred: &'a Nifti1Image<u8>,
    label: u8,
}

impl<'a> Evaluator<'a> {
    /// Creates a new instance of Evaluator from `nii::Nifti1Image<u8>`
    /// Example:
    ///
    /// ```rust
    /// use nii;
    /// use mikan::Evaluator;
    ///
    /// let gt = nii::read_image::<u8>(r"data\patients_26_ground_truth.nii.gz");
    /// let pred = nii::read_image::<u8>(r"data\patients_26_segmentation.nii.gz");
    /// let label = 1;
    /// let evaluator = Evaluator::new(&gt, &pred, label);
    ///
    /// let dsc = evaluator.get_dice();              // Dice Coefficient
    /// let hd = evaluator.get_hausdorff_distance(); // Hausdorff Distance
    ///
    /// ```
    pub fn new(gt: &'a Nifti1Image<u8>, pred: &'a Nifti1Image<u8>, label: u8) -> Evaluator<'a> {
        Evaluator {
            cm: ConfusionMatrix::new(gt, pred, label),
            dist: OnceCell::new(),
            gt,
            pred,
            label,
        }
    }

    /// Calculates Sensitivity
    ///
    /// Also known as:
    /// * TPR (True Positive Rate)
    /// * Recall
    /// * Hit Rate
    ///
    /// Implementation:
    /// $$\text{Sensitivity} = \dfrac{TP}{TP + FN}$$
    ///
    pub fn get_senstivity(&self) -> f64 {
        self.cm.get_senstivity()
    }

    /// Calculates Specificity
    ///
    /// Also known as:
    /// * TNR (True Negative Rate)
    /// * Selectivity
    ///
    /// Implementation:
    /// $$\text{Specificity} = \dfrac{TN}{TN + FP}$$
    ///
    pub fn get_specificity(&self) -> f64 {
        self.cm.get_specificity()
    }

    /// Calculates Precision
    ///
    /// Also known as:
    /// * PPV (Positive Predictive Value)
    ///
    /// Implementation:
    /// $$\text{Precision} = \dfrac{TP}{TP + FP}$$
    pub fn get_precision(&self) -> f64 {
        self.cm.get_precision()
    }

    /// Calculates Accuracy
    ///
    /// Also known as:
    /// * ACC
    /// * Rand Index (RI)
    ///
    /// Implementation:
    /// $$\text{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN}$$
    ///
    pub fn get_accuracy(&self) -> f64 {
        self.cm.get_accuracy()
    }

    /// Calculates Balanced Accuracy
    ///
    /// Also known as:
    /// * BACC
    /// * Balanced Classification Rate
    ///
    /// Implementation:
    /// $$\text{BACC} = \dfrac{\text{Sensitivity} + \text{Specificity}}{2}$$
    ///
    pub fn get_balanced_accuracy(&self) -> f64 {
        self.cm.get_balanced_accuracy()
    }

    /// Calculates Dice Coefficient
    ///
    /// Also known as:
    /// * DSC (Dice Similarity Coefficient)
    /// * F1-Score
    /// * Sørensen–Dice coefficient
    ///
    /// Implementation:
    /// $$\text{Dice} = \dfrac{2TP}{2TP + FP + FN}$$
    ///
    pub fn get_dice(&self) -> f64 {
        self.cm.get_dice()
    }

    /// Calculates F-Score
    ///
    /// Also known as:
    /// * F1-Score
    /// * F-measure
    /// * Harmonic mean of precision and recall
    ///
    /// Implementation:
    /// $$\text{F1} = \dfrac{2TP}{2TP + FP + FN}$$
    ///
    pub fn get_f_score(&self) -> f64 {
        self.cm.get_f_score()
    }

    /// Calculates F-beta Score
    ///
    /// # Arguments
    /// * `beta` - Weight of precision in harmonic mean
    ///
    /// Implementation:
    /// $$\text{F}_\beta = \dfrac{(1+\beta^2)TP}{(1+\beta^2)TP + \beta^2FN + FP}$$
    ///
    pub fn get_f_beta_score(&self, beta: u8) -> f64 {
        self.cm.get_f_beta_score(beta)
    }

    /// Calculates Jaccard Score
    ///
    /// Also known as:
    /// * IoU (Intersection over Union)
    /// * Jaccard Index
    /// * Jaccard Similarity Coefficient
    ///
    /// Implementation:
    /// $$\text{IoU} = \dfrac{TP}{TP + FP + FN}$$
    ///
    pub fn get_jaccard_score(&self) -> f64 {
        self.cm.get_jaccard_score()
    }

    /// Calculates False Negative Rate (FNR)
    ///
    /// Also known as:
    /// * Miss Rate
    /// * Type II error rate
    ///
    /// Implementation:
    /// $$\text{FNR} = \dfrac{FN}{FN + TP}$$
    ///
    pub fn get_fnr(&self) -> f64 {
        self.cm.get_fnr()
    }

    /// Calculates False Positive Rate (FPR)
    ///
    /// Also known as:
    /// * Fall-out
    /// * Type I error rate
    ///
    /// Implementation:
    /// $$\text{FPR} = \dfrac{FP}{FP + TN}$$
    ///
    pub fn get_fpr(&self) -> f64 {
        self.cm.get_fpr()
    }

    /// Calculates Volume Similarity
    ///
    /// Also known as:
    /// * VS
    /// * Volumetric Overlap Error
    ///
    /// Implementation:
    /// $$\text{VS} = 1 - \dfrac{|FN-FP|}{2TP + FP + FN}$$
    ///
    pub fn get_volume_similarity(&self) -> f64 {
        self.cm.get_volume_similarity()
    }

    /// Calculates Area Under the Curve
    ///
    /// Also known as:
    /// * AUC
    /// * AUC_trapezoid
    /// * ROC AUC (for binary classification)
    ///
    /// Implementation:
    /// $$\text{AUC} = 1 - \dfrac{FPR + FNR}{2}$$
    ///
    pub fn get_auc(&self) -> f64 {
        self.cm.get_auc()
    }

    /// Calculates Cohen's Kappa
    ///
    /// Also known as:
    /// * KAP
    /// * Kappa
    /// * Cohen's Kappa Coefficient
    ///
    /// Implementation:
    /// $$\text{Kappa} = \dfrac{fa - fc}{N - fc}$$
    ///
    /// where:
    /// - fa = TP + TN
    /// - fc = ((TN+FN)(TN+FP) + (FP+TP)(FN+TP))/N
    /// - N = Total samples
    ///
    pub fn get_kappa(&self) -> f64 {
        self.cm.get_kappa()
    }

    /// Calculates Matthews Correlation Coefficient
    ///
    /// Also known as:
    /// * MCC
    /// * Phi Coefficient
    /// * Matthews Phi Coefficient
    ///
    /// Implementation:
    /// $$\text{MCC} = \dfrac{TP\times TN - FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
    ///
    pub fn get_mcc(&self) -> f64 {
        self.cm.get_mcc()
    }

    /// Calculates Normalized Matthews Correlation Coefficient
    ///
    /// Also known as:
    /// * NMCC
    /// * Normalized MCC
    ///
    /// Implementation:
    /// $$\text{NMCC} = \dfrac{MCC + 1}{2}$$
    ///
    pub fn get_nmcc(&self) -> f64 {
        self.cm.get_nmcc()
    }

    /// Calculates Adjusted Matthews Correlation Coefficient
    ///
    /// Also known as:
    /// * AMCC
    /// * Adjusted MCC
    ///
    /// Implementation:
    /// $$\text{AMCC} = abs(MCC)$$
    ///
    pub fn get_amcc(&self) -> f64 {
        self.cm.get_amcc()
    }

    /// Calculates Adjusted Rand Index
    ///
    /// Also known as:
    /// * ARI
    /// * Adjusted Rand Score
    /// * Hubert and Arabie adjusted Rand index
    ///
    /// Implementation:
    /// $$\text{ARI} = \dfrac{2(TP\times TN - FP\times FN)}{(TP+FN)(FN+TN) + (TP+FP)(FP+TN)}$$
    ///
    pub fn get_adjust_rand_score(&self) -> f64 {
        self.cm.get_adjust_rand_score()
    }

    /// Returns the Distance instance, creating it if not already initialized
    fn get_dist(&self) -> &Distance {
        self.dist
            .get_or_init(|| Distance::new(self.gt, self.pred, self.label))
    }

    /// Calculates Hausdorff Distance
    ///
    /// [Implementation](https://metrics-reloaded.dkfz.de/metric?id=hd):
    /// $$ d(a,B) = \min_{b\in B}d(a,b) $$
    /// $$ HD(A,B) = \max(\max_{a\in A}d(a,B), \max_{b\in B}d(b,A)) $$
    ///
    pub fn get_hausdorff_distance(&self) -> f64 {
        self.get_dist().get_hausdorff_distance()
    }

    /// Calculates 95th percentile Hausdorff Distance
    ///
    /// [Implementation](https://metrics-reloaded.dkfz.de/metric?id=hd95):
    /// $$ d_{95}(A, B) = X_{95}\{\min_{b\in B}d(a,b)\} $$
    /// $$ HD_{95}(A, B) = \max \{ d_{95}(A, B), d_{95}(B, A) \} $$
    pub fn get_hausdorff_distance_95(&self) -> f64 {
        self.get_dist().get_hausdorff_distance_95()
    }

    /// Calculates Average Symmetric Surface Distance
    ///
    /// Also known as:
    /// * ASSD
    /// * Average Surface Distance
    ///
    /// [Implementation](https://metrics-reloaded.dkfz.de/metric?id=assd):
    /// $$ d(a,B) = \min_{b\in B}d(a,b) $$
    /// $$ ASSD(A,B) = \dfrac{\sum_{a\in A}d(a,B) + \sum_{b\in B}d(b,A)}{|A|+ |B|} $$
    ///
    pub fn get_assd(&self) -> f64 {
        self.get_dist().get_assd()
    }

    /// Calculates Maximum Average Surface Distance
    ///
    /// Also known as:
    /// * MASD
    /// * Maximum Surface Distance
    ///
    /// [Implementation](https://metrics-reloaded.dkfz.de/metric?id=masd):
    /// $$ d(a,B) = \min_{b\in B}d(a,b) $$
    /// $$ MASD(A,B) = \dfrac{1}{2}(\dfrac{\sum_{a\in A}d(a,B)}{|A|} + \dfrac{\sum_{b\in B}d(b,A)}{|B|})$$
    ///
    pub fn get_masd(&self) -> f64 {
        self.get_dist().get_masd()
    }

    /// Returns all confusion matrix based metrics as a BTreeMap
    pub fn get_cm_all(&self) -> BTreeMap<String, f64> {
        self.cm.get_all()
    }

    /// Returns all distance-based metrics as a BTreeMap
    pub fn get_dist_all(&self) -> BTreeMap<String, f64> {
        self.get_dist().get_all()
    }

    /// Returns all metrics (confusion matrix and distance-based) as a BTreeMap
    pub fn get_all(&self) -> BTreeMap<String, f64> {
        let mut map = self.cm.get_all();
        map.extend(self.get_dist().get_all());
        map
    }
}

/// A simple function api to calculate lots of metrics for given labels.
///
/// # Arguments
///
/// * `gt` - Ground truth image as a `Nifti1Image<u8>`
/// * `pred` - Predicted segmentation image as a `Nifti1Image<u8>`
/// * `labels` - Vector of label values to evaluate
/// * `with_distance` - Boolean flag to include distance-based metrics
///
/// # Examples
///
/// ```rust
/// use nii;
/// use mikan::metrics;
///
/// let gt = nii::read_image::<u8>(r"data\patients_26_ground_truth.nii.gz");
/// let pred = nii::read_image::<u8>(r"data\patients_26_segmentation.nii.gz");
///
/// // Calculate only confusion matrix based metrics
/// let basic_metrics = metrics(&gt, &pred, vec![1, 2, 3], false);
///
/// // Calculate both confusion matrix and distance based metrics
/// let all_metrics = metrics(&gt, &pred, vec![1, 2, 3], true);
/// ```
pub fn metrics(
    gt: &Nifti1Image<u8>,
    pred: &Nifti1Image<u8>,
    labels: Vec<u8>,
    with_distance: bool,
) -> Vec<BTreeMap<String, f64>> {
    let mut mat_results: Vec<BTreeMap<String, f64>> = labels
        .par_iter()
        .map(|&label| {
            let cm = ConfusionMatrix::new(&gt, &pred, label);
            let mut all_results = cm.get_all();
            all_results.insert("label".to_string(), label as f64);
            all_results
        })
        .collect();

    if with_distance {
        let dist_results: Vec<BTreeMap<String, f64>> = labels
            .par_iter()
            .map(|&label| {
                let cm = Distance::new(&gt, &pred, label);
                let mut all_results = cm.get_all();
                all_results.insert("label".to_string(), label as f64);
                all_results
            })
            .collect();
        for (map1, map2) in mat_results.iter_mut().zip(dist_results.iter()) {
            map1.extend(map2.iter().map(|(k, v)| (k.clone(), *v)));
        }
    }
    mat_results
}

/// A simple function api to calculate all metrics for all labels.
///
/// # Examples
///
/// ```rust
/// use nii;
/// use mikan::all;
///
/// let gt_image = nii::read_image::<u8>(r"data\patients_26_ground_truth.nii.gz");
/// let pred_image = nii::read_image::<u8>(r"data\patients_26_segmentation.nii.gz");
///
/// let metrics = all(&gt_image, &pred_image);
///
/// for metric in metrics.iter() {
///    println!("{:?}", metric);
/// }
/// ```
pub fn all(gt: &Nifti1Image<u8>, pred: &Nifti1Image<u8>) -> Vec<BTreeMap<String, f64>> {
    let labels = merge_vector(
        get_unique_labels_parallel(gt.ndarray()),
        get_unique_labels_parallel(pred.ndarray()),
        false,
    );
    metrics(gt, pred, labels, true)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::error::Error;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn test_metrics_wo_distances() -> Result<(), Box<dyn Error>> {
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);

        let results = metrics(&gt, &pred, vec![1, 2, 3, 4, 5], false);
        println!("{:?}", results);
        Ok(())
    }

    #[test]
    fn test_metrics_with_distances() -> Result<(), Box<dyn Error>> {
        let t = std::time::Instant::now();
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);
        println!("IO Cost {} ms", t.elapsed().as_millis());

        let t = std::time::Instant::now();
        let results = metrics(&gt, &pred, vec![1, 2, 3, 4, 5], true);
        println!("{:?}", results);
        println!("Calc Cost {} ms", t.elapsed().as_millis());

        Ok(())
    }

    #[test]
    fn test_api() -> Result<(), Box<dyn Error>> {
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);

        let t = Instant::now();

        let label = 1;
        let dist = Evaluator::new(&gt, &pred, label);

        let hd = dist.get_hausdorff_distance();
        println!("Cost {:?} ms", t.elapsed().as_millis());

        let hd95 = dist.get_hausdorff_distance_95();
        println!("Cost {:?} ms", t.elapsed().as_millis());

        let assd = dist.get_assd();
        println!("Cost {:?} ms", t.elapsed().as_millis());

        let masd = dist.get_masd();
        println!("Cost {:?} ms", t.elapsed().as_millis());

        let _cm = dist.get_cm_all();
        println!("Cost {:?} ms", t.elapsed().as_millis());

        let _all = dist.get_all();
        println!("Cost {:?} ms", t.elapsed().as_millis());

        println!("Hausdorff distance: {} mm", hd);
        println!("Hausdorff distance 95%: {} mm", hd95);
        println!("Average Symmetric Surface Distance: {} mm", assd);
        println!("Mean Average Surface Distance: {} mm", masd);
        println!("Cost {:?} ms", t.elapsed().as_millis());

        Ok(())
    }
}
