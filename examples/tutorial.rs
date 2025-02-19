use env_logger;
use mikan::Evaluator;
use nii;

fn main() {
    env_logger::init();
    let gt = nii::read_image::<u8>(r"data\patients_26_ground_truth.nii.gz");
    let pred = nii::read_image::<u8>(r"data\patients_26_segmentation.nii.gz");
    let label = 1;

    let evaluator = Evaluator::new(&gt, &pred, label);

    let dsc = evaluator.get_dice(); // Dice Coefficient
    let hd = evaluator.get_hausdorff_distance(); // Hausdorff Distance

    println!("Dice Coefficient: {}", dsc);
    println!("Hausdorff Distance: {}", hd);

    let all = evaluator.get_all();

    // Print all metrics
    for (k, v) in all.iter() {
        println!("{}: {}", k, v);
    }
}
