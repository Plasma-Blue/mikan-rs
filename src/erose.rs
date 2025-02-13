use ndarray::{arr3, Array3};
use ndarray_ndimage::binary_erosion;

/// u8 -> bool
pub(crate) fn binary_erosion_u8(
    mask: &Array3<u8>,
    kernel: &Array3<u8>,
    iterations: usize,
) -> Array3<u8> {
    // u8 -> bool
    let mask_bool = mask.mapv(|x| x != 0);
    let kernel_bool = kernel.mapv(|x| x != 0);

    let result_bool = binary_erosion(&mask_bool, &kernel_bool, iterations);

    // bool -> u8
    result_bool.mapv(|x| x as u8)
}

pub(crate) fn get_binary_edge(mask: &Array3<u8>) -> Array3<u8> {
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
mod tests {
    use super::{binary_erosion_u8, get_binary_edge};
    use ndarray::arr3;
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
