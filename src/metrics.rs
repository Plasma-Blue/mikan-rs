use crate::cm::ConfusionMatrix;
use crate::kdtree::Distance;
use crate::utils::{get_unique_labels_parallel, merge_vector};
use nii::Nifti1Image;
use rayon::prelude::*;
use std::collections::BTreeMap;

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
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);

        let results = metrics(gt, pred, vec![1, 2, 3, 4, 5], true);
        println!("{:?}", results);
        Ok(())
    }
}
