use nii::Nifti1Image;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    tp_count: u32,
    tn_count: u32,
    fp_count: u32,
    fn_count: u32,
}

impl ConfusionMatrix {
    /// init ConfusionMatrix using tp/tn/fp/fn directly, can be used for binary classification task
    pub fn set(tp_count: u32, tn_count: u32, fp_count: u32, fn_count: u32) -> Self {
        ConfusionMatrix {
            tp_count,
            tn_count,
            fp_count,
            fn_count,
        }
    }

    /// init ConfusionMatrix using two segmentation mask, can be used for segmentation task
    pub fn new(gt: &Nifti1Image<u8>, pred: &Nifti1Image<u8>, label: u8) -> Self {
        let gt_arr = gt.ndarray();
        let pred_arr = pred.ndarray();

        let mut tp_count = 0;
        let mut fp_count = 0;
        let mut fn_count = 0;
        let mut tn_count = 0;

        for (&a, &b) in gt_arr.iter().zip(pred_arr.iter()) {
            if a == label && b == label {
                tp_count += 1;
            } else if a != label && b == label {
                fp_count += 1;
            } else if a == label && b != label {
                fn_count += 1;
            } else if a != label && b != label {
                tn_count += 1;
            }
        }
        ConfusionMatrix {
            tp_count,
            fp_count,
            fn_count,
            tn_count,
        }
    }

    /// Recall/Sensitivity/Hit rate/True positive rate (TPR)/敏感性/召回率
    pub fn get_senstivity(&self) -> f64 {
        (self.tp_count) as f64 / (self.tp_count + self.fn_count) as f64
    }

    /// Selectivity/Specificity/True negative rate (TNR)/特异性
    pub fn get_specificity(&self) -> f64 {
        (self.tn_count) as f64 / (self.tn_count + self.fp_count) as f64
    }

    /// Precision/Positive predictive value (PPV)/精确性
    pub fn get_precision(&self) -> f64 {
        (self.tp_count) as f64 / (self.tp_count + self.fp_count) as f64
    }
    /// accuracy/acc/Rand Index/RI/准确性
    pub fn get_accuracy(&self) -> f64 {
        (self.tp_count + self.tn_count) as f64
            / (self.tp_count + self.tn_count + self.fp_count + self.fn_count) as f64
    }

    /// balanced accuracy / BACC
    pub fn get_balanced_accuracy(&self) -> f64 {
        (self.get_senstivity() + self.get_specificity()) / 2.0
    }

    /// Dice/DSC
    pub fn get_dice(&self) -> f64 {
        (2 * self.tp_count) as f64 / (2 * self.tp_count + self.fp_count + self.fn_count) as f64
    }

    /// f-score
    pub fn get_f_score(&self) -> f64 {
        (2 * self.tp_count) as f64 / (2 * self.tp_count + self.fp_count + self.fn_count) as f64
    }

    /// f-beta score
    pub fn get_f_beta_score(&self, beta: u8) -> f64 {
        let beta = beta as u32;
        ((1 + beta.pow(2)) * self.tp_count) as f64
            / ((1 + beta.pow(2)) * self.tp_count + beta.pow(2) * self.fn_count * self.fp_count)
                as f64
    }

    /// jaccard score/IoU
    pub fn get_jaccard_score(&self) -> f64 {
        (self.tp_count) as f64 / (self.tp_count + self.fp_count + self.fn_count) as f64
    }

    /// fnr
    pub fn get_fnr(&self) -> f64 {
        self.fn_count as f64 / (self.fn_count + self.tp_count) as f64
    }

    /// fpr
    pub fn get_fpr(&self) -> f64 {
        self.fp_count as f64 / (self.fp_count + self.tn_count) as f64
    }

    /// volume similarity/VS/体积相似性
    pub fn get_volume_similarity(&self) -> f64 {
        1.0 - (self.fn_count as i32 - self.fp_count as i32).abs() as f64
            / (2 * self.tp_count + self.fp_count + self.fn_count) as f64
    }

    /// AUC/AUC_trapezoid/binary label AUC
    pub fn get_auc(&self) -> f64 {
        1.0 - 0.5 * (self.get_fpr() + self.get_fnr())
    }

    /// KAP/Kappa/CohensKapp
    pub fn get_kappa(&self) -> f64 {
        let sum_ = self.tp_count as f64
            + self.tn_count as f64
            + self.fp_count as f64
            + self.fn_count as f64;
        let fa = self.tp_count as f64 + self.tn_count as f64;
        let fc = ((self.tn_count as f64 + self.fn_count as f64)
            * (self.tn_count as f64 + self.fp_count as f64)
            + (self.fp_count as f64 + self.tp_count as f64)
                * (self.fn_count as f64 + self.tp_count as f64))
            / sum_;
        (fa - fc) / (sum_ - fc)
    }

    pub fn get_mcc(&self) -> f64 {
        let top = self.tp_count as f64 * self.tn_count as f64
            - self.fp_count as f64 * self.fn_count as f64;

        // very huge
        let bot_raw = (self.tp_count as f64 + self.fp_count as f64)
            * (self.tp_count as f64 + self.fn_count as f64)
            * (self.tn_count as f64 + self.fp_count as f64)
            * (self.tn_count as f64 + self.fn_count as f64);
        let bot = bot_raw.sqrt();
        top / bot
    }

    pub fn get_nmcc(&self) -> f64 {
        let mcc = self.get_mcc();
        (mcc + 1.0) / 2.0
    }

    pub fn get_amcc(&self) -> f64 {
        self.get_mcc().abs()
    }

    /// adjust rand score/adjust rand index/ARI
    pub fn get_adjust_rand_score(&self) -> f64 {
        let top = self.tp_count as f64 * self.tn_count as f64
            - self.fp_count as f64 * self.fn_count as f64;
        let bot = (self.tp_count as f64 + self.fn_count as f64)
            * (self.fn_count as f64 + self.tn_count as f64)
            + (self.tp_count as f64 + self.fp_count as f64)
                * (self.fp_count as f64 + self.tn_count as f64);
        2.0 * top / bot
    }

    pub fn get_all(&self) -> BTreeMap<String, f64> {
        let mut map = BTreeMap::new();
        map.insert("tp".to_string(), self.tp_count as f64);
        map.insert("tn".to_string(), self.tn_count as f64);
        map.insert("fp".to_string(), self.fp_count as f64);
        map.insert("fn".to_string(), self.fn_count as f64);
        map.insert("senstivity".to_string(), self.get_senstivity());
        map.insert("specificity".to_string(), self.get_specificity());
        map.insert("precision".to_string(), self.get_precision());
        map.insert("accuracy".to_string(), self.get_accuracy());
        map.insert(
            "balanced_accuracy".to_string(),
            self.get_balanced_accuracy(),
        );
        map.insert("dice".to_string(), self.get_dice());
        map.insert("f_score".to_string(), self.get_f_score());
        map.insert("jaccard_score".to_string(), self.get_jaccard_score());
        map.insert("fnr".to_string(), self.get_fnr());
        map.insert("fpr".to_string(), self.get_fpr());
        map.insert(
            "volume_similarity".to_string(),
            self.get_volume_similarity(),
        );
        map.insert("auc".to_string(), self.get_auc());
        map.insert("kappa".to_string(), self.get_kappa());
        map.insert("mcc".to_string(), self.get_mcc());
        map.insert("nmcc".to_string(), self.get_nmcc());
        map.insert("amcc".to_string(), self.get_amcc());
        map.insert(
            "adjust_rand_score".to_string(),
            self.get_adjust_rand_score(),
        );
        map
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use rayon::prelude::*;
    use std::error::Error;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn test_matrix_from_image() -> Result<(), Box<dyn Error>> {
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);

        let t = Instant::now();

        // let unique_labels = merge_vector(unique(&gt_arr), unique(&pred_arr), true);
        let unique_labels = [1, 2, 3, 4, 5];

        let results: Vec<BTreeMap<String, f64>> = unique_labels
            .par_iter()
            .map(|&label| {
                let cm = ConfusionMatrix::new(&gt, &pred, label);
                let mut all_results = cm.get_all();
                all_results.insert("label".to_string(), label as f64);
                all_results
            })
            .collect();
        println!("{:?}", results);
        println!("Cost {:?} ms", t.elapsed().as_millis());

        Ok(())
    }

    #[test]
    fn test_matrix_from_direct() -> Result<(), Box<dyn Error>> {
        let cm = ConfusionMatrix::set(1, 2, 3, 4);
        let all_results = cm.get_all();
        println!("{:?}", all_results);
        Ok(())
    }
}
