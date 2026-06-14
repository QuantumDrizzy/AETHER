use pyo3::prelude::*;
use pyo3::types::PyDict;
use aether_core::material::Material;

#[pyclass]
pub struct PyMaterial {
    inner: Material,
}

#[pymethods]
impl PyMaterial {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
}

#[pyclass]
pub struct PyAetherDb {
    // Stub wrapper around AetherDb
}

#[pymethods]
impl PyAetherDb {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

#[pymodule]
fn aether_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyAetherDb>()?;
    Ok(())
}
