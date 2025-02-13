use std::collections::BTreeMap;

use crate::erose::get_binary_edge;
use crate::utils::{argwhere, get_percentile, mean};
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use nii::Nifti1Image;
use rayon::prelude::*;

struct KDTree {
    tree: KdTree<f32, usize, [f32; 3]>,
}

impl KDTree {
    fn new(points: &[(f32, f32, f32)]) -> Self {
        let mut kdtree = KdTree::new(3);
        for (idx, p) in points.iter().enumerate() {
            let point = [p.0 as f32, p.1 as f32, p.2 as f32];
            kdtree.add(point, idx).unwrap();
        }
        KDTree { tree: kdtree }
    }

    fn query(&self, points: &[(f32, f32, f32)]) -> Vec<f32> {
        points
            .par_iter()
            .map(|p| {
                let point = [p.0 as f32, p.1 as f32, p.2 as f32];
                let a = self.tree.nearest(&point, 1, &squared_euclidean).unwrap()[0];
                a.0 as f32
            })
            .collect()
    }
}

pub struct Distance {
    dist_pred_to_gt: Vec<f32>,
    dist_gt_to_pred: Vec<f32>,
}

impl Distance {
    pub fn new(gt: &Nifti1Image<u8>, pred: &Nifti1Image<u8>, label: u8) -> Self {
        // TODO: support different size, spacing, direction in the future, now we assume they are the same
        // Actually, having gt and pred in the same world space is enough
        assert_eq!(gt.get_size(), pred.get_size(), "Size mismatch");
        assert_eq!(gt.get_spacing(), pred.get_spacing(), "Spacing mismatch");
        assert_eq!(
            gt.get_direction(),
            pred.get_direction(),
            "Direction mismatch"
        );

        let spacing = gt.get_spacing();

        let gt_arr = gt.ndarray();
        let pred_arr = pred.ndarray();

        // Binarize
        let gt_arr = gt_arr.mapv(|x| if x == label { 1 } else { 0 });
        let pred_arr = pred_arr.mapv(|x| if x == label { 1 } else { 0 });

        // Get edge
        let gt_edge = get_binary_edge(&gt_arr);
        let pred_edge = get_binary_edge(&pred_arr);

        // Get edge argwhere
        let gt_argw: Vec<(usize, usize, usize)> = argwhere(&gt_edge, 1); // (z,y,x)
        let pred_argw: Vec<(usize, usize, usize)> = argwhere(&pred_edge, 1);

        // Convert to physical coordinates
        let gt_argw: Vec<(f32, f32, f32)> = gt_argw
            .par_iter()
            .map(|x| {
                let z = x.0 as f32 * spacing[2];
                let y = x.1 as f32 * spacing[1];
                let x = x.2 as f32 * spacing[0];
                (z, y, x)
            })
            .collect();

        let pred_argw: Vec<(f32, f32, f32)> = pred_argw
            .par_iter()
            .map(|x| {
                let z = x.0 as f32 * spacing[2];
                let y = x.1 as f32 * spacing[1];
                let x = x.2 as f32 * spacing[0];
                (z, y, x)
            })
            .collect();

        let dist_pred_to_gt = KDTree::new(&gt_argw).query(&pred_argw);
        let dist_gt_to_pred = KDTree::new(&pred_argw).query(&gt_argw);

        let dist_pred_to_gt = dist_pred_to_gt.par_iter().map(|x| x.sqrt()).collect(); // square
        let dist_gt_to_pred = dist_gt_to_pred.par_iter().map(|x| x.sqrt()).collect(); // square

        Distance {
            dist_pred_to_gt,
            dist_gt_to_pred,
        }
    }
    pub fn get_hausdorff_distance_95(&self) -> f32 {
        let mut dist_pred_to_gt = self.dist_pred_to_gt.clone();
        let mut dist_gt_to_pred = self.dist_gt_to_pred.clone();
        f32::max(
            get_percentile(&mut dist_pred_to_gt, 0.95),
            get_percentile(&mut dist_gt_to_pred, 0.95),
        )
    }

    pub fn get_hausdorff_distance(&self) -> f32 {
        f32::max(
            Array::from(self.dist_pred_to_gt.clone())
                .max()
                .unwrap()
                .clone(),
            Array::from(self.dist_gt_to_pred.clone())
                .max()
                .unwrap()
                .clone(),
        )
    }

    pub fn get_assd(&self) -> f32 {
        let merged = self
            .dist_pred_to_gt
            .iter()
            .chain(self.dist_gt_to_pred.iter())
            .cloned()
            .collect();
        mean(&merged)
    }

    pub fn get_masd(&self) -> f32 {
        (mean(&self.dist_pred_to_gt) + mean(&self.dist_gt_to_pred)) / 2.0
    }

    pub fn get_all(&self) -> BTreeMap<String, f64> {
        let mut results = BTreeMap::new();
        results.insert(
            "hausdorff_distance".to_string(),
            self.get_hausdorff_distance() as f64,
        );
        results.insert(
            "hausdorff_distance_95".to_string(),
            self.get_hausdorff_distance_95() as f64,
        );
        results.insert("assd".to_string(), self.get_assd() as f64);
        results.insert("masd".to_string(), self.get_masd() as f64);
        results
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::error::Error;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn test_distances() -> Result<(), Box<dyn Error>> {
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);

        let t = Instant::now();

        let label = 1;
        let dist = Distance::new(&gt, &pred, label);

        let hd = dist.get_hausdorff_distance();
        let hd95 = dist.get_hausdorff_distance_95();
        let assd = dist.get_assd();
        let masd = dist.get_masd();

        println!("Hausdorff distance: {} mm", hd);
        println!("Hausdorff distance 95%: {} mm", hd95);
        println!("Average Symmetric Surface Distance: {} mm", assd);
        println!("Mean Average Surface Distance: {} mm", masd);
        println!("Cost {:?} ms", t.elapsed().as_millis());

        Ok(())
    }

    #[test]
    fn test_mp_distances() -> Result<(), Box<dyn Error>> {
        let gt = Path::new(r"data\patients_26_ground_truth.nii.gz");
        let pred = Path::new(r"data\patients_26_segmentation.nii.gz");

        let gt = nii::read_image::<u8>(gt);
        let pred = nii::read_image::<u8>(pred);

        let t = Instant::now();

        let label: Vec<u8> = vec![1, 2, 3, 4, 5];

        let results: Vec<f32> = label
            .par_iter()
            .map(|label| {
                let dist = Distance::new(&gt, &pred, *label);
                dist.get_hausdorff_distance_95()
            })
            .collect();

        println!("Hausdorff distance 95: {:?} mm", results);
        println!("Cost {:?} ms", t.elapsed().as_millis());

        Ok(())
    }
}
