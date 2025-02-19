use chrono;
use colored::*;
use env_logger::Builder;
use log::LevelFilter;
use ndarray::Zip;
use ndarray::{arr3, Array3, ArrayView3};
use ndarray_ndimage::binary_erosion;
use rayon::prelude::*;
use std::collections::HashSet;
use std::hash::Hash;
use std::io::Write;

pub fn init_logger() {
    Builder::new()
        .format(|buf, record| {
            let (level, message_color) = match record.level() {
                log::Level::Error => ("ERROR".red(), Color::Red),
                log::Level::Warn => ("WARN".yellow(), Color::Yellow),
                log::Level::Info => ("INFO".green(), Color::Green),
                log::Level::Debug => ("DEBUG".blue(), Color::Blue),
                log::Level::Trace => ("TRACE".purple(), Color::Magenta),
            };
            let time = chrono::Local::now()
                .format("%Y-%m-%d %H:%M:%S%.3f")
                .to_string()
                .truecolor(140, 194, 92);
            let location = format!(
                "{}:{}",
                record.module_path().unwrap_or("unknown"),
                record.line().unwrap_or(0)
            )
            .truecolor(66, 179, 184);
            let message = record.args().to_string().color(message_color);
            writeln!(buf, "{} | {} | {} | {}", time, level, location, message)
        })
        .filter(None, LevelFilter::Warn) // default: warning
        .init();
}

fn partition(arr: &mut [f32], low: usize, high: usize) -> usize {
    let pivot = arr[high];
    let mut i = low;
    for j in low..high {
        if arr[j] < pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, high);
    i
}

pub fn get_percentile(arr: &mut [f32], percentile: f32) -> f32 {
    fn quickselect_helper(arr: &mut [f32], low: usize, high: usize, k: usize) -> f32 {
        if low == high {
            return arr[low];
        }
        let pivot_index = partition(arr, low, high);
        if k == pivot_index {
            arr[k]
        } else if k < pivot_index {
            quickselect_helper(arr, low, pivot_index - 1, k)
        } else {
            quickselect_helper(arr, pivot_index + 1, high, k)
        }
    }
    let percentile = (arr.len() as f32 * percentile).round() as usize - 1;
    quickselect_helper(arr, 0, arr.len() - 1, percentile)
}

pub fn mean(data: &Vec<f32>) -> f32 {
    let sum: f32 = data.iter().sum();
    let count = data.len();
    sum / count as f32
}

pub fn get_unique_labels_parallel(array: ArrayView3<u8>) -> Vec<u8> {
    let chunks = array.as_slice().expect("Contiguous array").par_chunks(4096);
    let presents: Vec<_> = chunks
        .map(|chunk| {
            let mut present = [false; 256];
            chunk.iter().for_each(|&x| present[x as usize] = true);
            present
        })
        .collect();

    let mut merged = [false; 256];
    for present in presents {
        for (i, &p) in present.iter().enumerate() {
            merged[i] |= p;
        }
    }

    merged
        .iter()
        .enumerate()
        .filter(|(_, &p)| p)
        .map(|(i, _)| i as u8)
        .collect()
}

pub fn merge_vector<T>(vec1: Vec<T>, vec2: Vec<T>, no_zero: bool) -> Vec<T>
where
    T: Eq + Hash + Ord + Default,
{
    let mut set: HashSet<T> = HashSet::new();
    set.extend(vec1);
    set.extend(vec2);
    let mut vec: Vec<T> = set.into_iter().collect();
    vec.sort();

    if no_zero {
        vec.retain(|x| *x != T::default());
    }
    vec
}

pub fn argwhere<T>(array: &Array3<T>, condition: T) -> Vec<(usize, usize, usize)>
where
    T: PartialEq,
{
    let mut indices = Vec::new();
    Zip::indexed(array).for_each(|(i, j, k), value| {
        if *value == condition {
            indices.push((i, j, k));
        }
    });
    indices
}

/// u8 -> bool
pub fn binary_erosion_u8(mask: &Array3<u8>, kernel: &Array3<u8>, iterations: usize) -> Array3<u8> {
    // u8 -> bool
    let mask_bool = mask.mapv(|x| x != 0);
    let kernel_bool = kernel.mapv(|x| x != 0);

    let result_bool = binary_erosion(&mask_bool, &kernel_bool, iterations);

    // bool -> u8
    result_bool.mapv(|x| x as u8)
}

pub fn get_binary_edge(mask: &Array3<u8>) -> Array3<u8> {
    let iterations = 1;
    let kernel = arr3(&[
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]);
    let result = binary_erosion_u8(&mask, &kernel, iterations);
    mask - result
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::arr3;
    use rand::Rng;
    use std::error::Error;

    fn generate_large_vec(size: usize) -> Vec<f32> {
        let mut rng = rand::rng();
        (0..size).map(|_| rng.random_range(0.0..100.0)).collect()
    }

    #[test]
    fn test_get_percentile() -> Result<(), Box<dyn Error>> {
        use std::time::Instant;

        let t = Instant::now();
        let mut data = generate_large_vec(100000);
        let percentile = get_percentile(&mut data, 0.95);
        println!("Time cost: {:?} ms", t.elapsed().as_millis());
        println!("95th percentile: {}", percentile);
        Ok(())
    }

    #[test]
    fn test_get_percentile_2() -> Result<(), Box<dyn Error>> {
        use rand::rng;
        use rand::seq::SliceRandom;
        use std::time::Instant;
        let mut data: Vec<f32> = (1..=100000000).map(|x| x as f32).collect();

        let mut rng = rng();
        data.shuffle(&mut rng);

        let t = Instant::now();
        let percentile = get_percentile(&mut data, 0.95);
        println!("Time cost: {:?} ms", t.elapsed().as_millis());
        println!("95th percentile: {}", percentile);
        Ok(())
    }

    #[test]
    fn test_mean() -> Result<(), Box<dyn Error>> {
        use std::time::Instant;

        let data = generate_large_vec(100000);
        let t = Instant::now();
        println!("Mean: {:?}", mean(&data));
        println!("Time cost: {:?} ms", t.elapsed().as_millis());

        Ok(())
    }

    #[test]
    fn test_binary_erosion_single_iteration() {
        let mask = arr3(&[
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]);

        let kernel = arr3(&[
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]);

        let result = binary_erosion_u8(&mask, &kernel, 1);

        let expected = arr3(&[
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_get_binary_edge() {
        let mask = arr3(&[
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]);

        let result = get_binary_edge(&mask);

        let expected = arr3(&[
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]);
        assert_eq!(result, expected);
    }
}
