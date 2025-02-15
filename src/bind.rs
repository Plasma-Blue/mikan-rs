use crate::api::metrics;
use crate::metrics::{ConfusionMatrix, Distance};
use crate::utils::{get_unique_labels_parallel, merge_vector};
use nii;
use pyo3::prelude::*;
use std::collections::BTreeMap;

#[pyclass]
pub struct ConfusionMatrixBind {
    inner: ConfusionMatrix,
}

#[pyclass]
pub struct DistanceBind {
    inner: Distance,
}

#[pymethods]
impl ConfusionMatrixBind {
    #[staticmethod]
    pub fn new(gt_pth: &str, pred_pth: &str, label: u8) -> PyResult<Self> {
        let gt = nii::read_image::<u8>(gt_pth);
        let pred = nii::read_image::<u8>(pred_pth);
        let inner = ConfusionMatrix::new(&gt, &pred, label);
        Ok(ConfusionMatrixBind { inner })
    }

    /// Recall/Sensitivity/Hit rate/True positive rate (TPR)/敏感性/召回率
    pub fn get_senstivity(&self) -> f64 {
        self.inner.get_senstivity()
    }

    /// Selectivity/Specificity/True negative rate (TNR)/特异性
    pub fn get_specificity(&self) -> f64 {
        self.inner.get_specificity()
    }

    /// Precision/Positive predictive value (PPV)/精确性
    pub fn get_precision(&self) -> f64 {
        self.inner.get_precision()
    }

    /// accuracy/acc/Rand Index/RI/准确性
    pub fn get_accuracy(&self) -> f64 {
        self.inner.get_accuracy()
    }

    /// balanced accuracy / BACC
    pub fn get_balanced_accuracy(&self) -> f64 {
        self.inner.get_balanced_accuracy()
    }

    /// Dice/DSC
    pub fn get_dice(&self) -> f64 {
        self.inner.get_dice()
    }

    /// f-score
    pub fn get_f_score(&self) -> f64 {
        self.inner.get_f_score()
    }

    /// f-beta score
    pub fn get_f_beta_score(&self, beta: u8) -> f64 {
        self.inner.get_f_beta_score(beta)
    }

    /// jaccard score/IoU
    pub fn get_jaccard_score(&self) -> f64 {
        self.inner.get_jaccard_score()
    }

    /// fnr
    pub fn get_fnr(&self) -> f64 {
        self.inner.get_fnr()
    }

    /// fpr
    pub fn get_fpr(&self) -> f64 {
        self.inner.get_fpr()
    }

    /// volume similarity/VS/体积相似性
    pub fn get_volume_similarity(&self) -> f64 {
        self.inner.get_volume_similarity()
    }

    /// AUC/AUC_trapezoid/binary label AUC
    pub fn get_auc(&self) -> f64 {
        self.inner.get_auc()
    }

    /// KAP/Kappa/CohensKapp
    pub fn get_kappa(&self) -> f64 {
        self.inner.get_kappa()
    }

    pub fn get_mcc(&self) -> f64 {
        self.inner.get_mcc()
    }

    pub fn get_nmcc(&self) -> f64 {
        self.inner.get_nmcc()
    }

    pub fn get_amcc(&self) -> f64 {
        self.inner.get_amcc()
    }

    /// adjust rand score/adjust rand index/ARI
    pub fn get_adjust_rand_score(&self) -> f64 {
        self.inner.get_adjust_rand_score()
    }

    pub fn get_all(&self) -> BTreeMap<String, f64> {
        self.inner.get_all()
    }
}

#[pymethods]
impl DistanceBind {
    #[staticmethod]
    pub fn new(gt_pth: &str, pred_pth: &str, label: u8) -> PyResult<Self> {
        let gt = nii::read_image::<u8>(gt_pth);
        let pred = nii::read_image::<u8>(pred_pth);
        let inner = Distance::new(&gt, &pred, label);
        Ok(DistanceBind { inner })
    }

    pub fn get_hausdorff_distance_95(&self) -> f32 {
        self.inner.get_hausdorff_distance_95()
    }

    pub fn get_hausdorff_distance(&self) -> f32 {
        self.inner.get_hausdorff_distance()
    }

    pub fn get_assd(&self) -> f32 {
        self.inner.get_assd()
    }

    pub fn get_masd(&self) -> f32 {
        self.inner.get_masd()
    }

    pub fn get_all(&self) -> BTreeMap<String, f64> {
        self.inner.get_all()
    }
}

#[pyfunction]
pub fn metrics_bind(
    gt_pth: &str,
    pred_pth: &str,
    labels: Vec<u8>,
    with_distances: bool,
) -> PyResult<Vec<BTreeMap<String, f64>>> {
    let gt = nii::read_image::<u8>(gt_pth);
    let pred = nii::read_image::<u8>(pred_pth);
    Ok(metrics(gt, pred, labels, with_distances))
}

#[pyfunction]
pub fn metrics_all_bind(gt_pth: &str, pred_pth: &str) -> PyResult<Vec<BTreeMap<String, f64>>> {
    let gt = nii::read_image::<u8>(gt_pth);
    let pred = nii::read_image::<u8>(pred_pth);
    let labels = merge_vector(
        get_unique_labels_parallel(gt.ndarray()),
        get_unique_labels_parallel(pred.ndarray()),
        false,
    );
    Ok(metrics(gt, pred, labels, true))
}

#[pymodule]
fn _mikan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConfusionMatrixBind>()?;
    m.add_class::<DistanceBind>()?;
    m.add_function(wrap_pyfunction!(metrics_all_bind, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_bind, m)?)?;
    Ok(())
}
