pub mod cm;
pub mod erose;
pub mod kdtree;
pub mod metrics;
pub mod utils;

pub use cm::ConfusionMatrix;
pub use kdtree::Distance;
pub use metrics::*;

mod bind;
use bind::*;
use pyo3::prelude::*;

#[pymodule]
fn _mikan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConfusionMatrixBind>()?;
    m.add_class::<DistanceBind>()?;
    m.add_function(wrap_pyfunction!(metrics_all_bind, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_bind, m)?)?;
    Ok(())
}
