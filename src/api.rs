use crate::metrics::{ConfusionMatrix, Distance};
use crate::utils::{get_unique_labels_parallel, merge_vector};
use std::collections::BTreeMap;

use nii::Nifti1Image;
use once_cell::unsync::OnceCell;
use rayon::prelude::*;

pub struct Evaluator<'a> {
    cm: ConfusionMatrix,
    dist: OnceCell<Distance>,
    gt: &'a Nifti1Image<u8>,
    pred: &'a Nifti1Image<u8>,
    label: u8,
}

impl<'a> Evaluator<'a> {
    pub fn new(gt: &'a Nifti1Image<u8>, pred: &'a Nifti1Image<u8>, label: u8) -> Evaluator<'a> {
        Evaluator {
            cm: ConfusionMatrix::new(gt, pred, label),
            dist: OnceCell::new(),
            gt,
            pred,
            label,
        }
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// Recall/Sensitivity/Hit rate/True positive rate (TPR)/敏感性/召回率
    ///
    /// $$\text{Sens} = \dfrac{TP}{TP+FN}$$
    ///
    pub fn get_senstivity(&self) -> f64 {
        self.cm.get_senstivity()
    }

    /// Selectivity/Specificity/True negative rate (TNR)/特异性
    pub fn get_specificity(&self) -> f64 {
        self.cm.get_specificity()
    }

    /// Precision/Positive predictive value (PPV)/精确性
    pub fn get_precision(&self) -> f64 {
        self.cm.get_precision()
    }

    /// accuracy/acc/Rand Index/RI/准确性
    pub fn get_accuracy(&self) -> f64 {
        self.cm.get_accuracy()
    }

    /// balanced accuracy / BACC
    pub fn get_balanced_accuracy(&self) -> f64 {
        self.cm.get_balanced_accuracy()
    }

    /// Dice/DSC
    pub fn get_dice(&self) -> f64 {
        self.cm.get_dice()
    }

    /// f-score
    pub fn get_f_score(&self) -> f64 {
        self.cm.get_f_score()
    }

    /// f-beta score
    pub fn get_f_beta_score(&self, beta: u8) -> f64 {
        self.cm.get_f_beta_score(beta)
    }

    /// jaccard score/IoU
    pub fn get_jaccard_score(&self) -> f64 {
        self.cm.get_jaccard_score()
    }

    /// fnr
    pub fn get_fnr(&self) -> f64 {
        self.cm.get_fnr()
    }

    /// fpr
    pub fn get_fpr(&self) -> f64 {
        self.cm.get_fpr()
    }

    /// volume similarity/VS/体积相似性
    pub fn get_volume_similarity(&self) -> f64 {
        self.cm.get_volume_similarity()
    }

    /// AUC/AUC_trapezoid/binary label AUC
    pub fn get_auc(&self) -> f64 {
        self.cm.get_auc()
    }

    /// KAP/Kappa/CohensKapp
    pub fn get_kappa(&self) -> f64 {
        self.cm.get_kappa()
    }

    /// mcc/MCC/Matthews correlation coefficient
    pub fn get_mcc(&self) -> f64 {
        self.cm.get_mcc()
    }

    /// nmcc/normalized mcc
    pub fn get_nmcc(&self) -> f64 {
        self.cm.get_nmcc()
    }

    /// amcc/adjusted mcc
    pub fn get_amcc(&self) -> f64 {
        self.cm.get_amcc()
    }

    /// adjust rand score/adjust rand index/ARI
    pub fn get_adjust_rand_score(&self) -> f64 {
        self.cm.get_adjust_rand_score()
    }

    pub fn cm_get_all(&self) -> BTreeMap<String, f64> {
        self.cm.get_all()
    }

    fn get_dist(&self) -> &Distance {
        self.dist
            .get_or_init(|| Distance::new(self.gt, self.pred, self.label))
    }

    pub fn get_hausdorff_distance_95(&self) -> f32 {
        self.get_dist().get_hausdorff_distance_95()
    }

    pub fn get_hausdorff_distance(&self) -> f32 {
        self.get_dist().get_hausdorff_distance()
    }

    pub fn get_assd(&self) -> f32 {
        self.get_dist().get_assd()
    }

    pub fn get_masd(&self) -> f32 {
        self.get_dist().get_masd()
    }

    pub fn get_dist_all(&self) -> BTreeMap<String, f64> {
        self.get_dist().get_all()
    }

    pub fn get_all(&self) -> BTreeMap<String, f64> {
        let mut map = self.cm.get_all();
        map.extend(self.get_dist().get_all());
        map.insert("label".to_string(), self.label as f64);
        map
    }
}

pub fn metrics(
    gt: Nifti1Image<u8>,
    pred: Nifti1Image<u8>,
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

pub fn metrics_all(gt: Nifti1Image<u8>, pred: Nifti1Image<u8>) -> Vec<BTreeMap<String, f64>> {
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

        let results = metrics(gt, pred, vec![1, 2, 3, 4, 5], false);
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
        let results = metrics(gt, pred, vec![1, 2, 3, 4, 5], true);
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

        let _cm = dist.cm_get_all();
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
