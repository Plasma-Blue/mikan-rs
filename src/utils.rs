use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use std::collections::HashSet;
use std::hash::Hash;

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

pub(crate) fn get_percentile(arr: &mut [f32], percentile: f32) -> f32 {
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

pub(crate) fn mean(data: &Vec<f32>) -> f32 {
    let sum: f32 = data.iter().sum();
    let count = data.len();
    sum / count as f32
}

pub(crate) fn get_unique_labels_parallel(array: &Array3<u8>) -> Vec<u8> {
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

pub(crate) fn merge_vector<T>(vec1: Vec<T>, vec2: Vec<T>, no_zero: bool) -> Vec<T>
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

pub(crate) fn argwhere<T>(array: &Array3<T>, condition: T) -> Vec<(usize, usize, usize)>
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
#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;
    use std::error::Error;

    fn generate_large_vec(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range(0.0..100.0)).collect()
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
    fn test_mean() -> Result<(), Box<dyn Error>> {
        use std::time::Instant;

        let data = generate_large_vec(100000);
        let t = Instant::now();
        println!("Mean: {:?}", mean(&data));
        println!("Time cost: {:?} ms", t.elapsed().as_millis());

        Ok(())
    }
}
