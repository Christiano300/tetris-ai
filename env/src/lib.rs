use pyo3::prelude::pymodule;

mod cbor;
mod environment;
mod tetris;
mod util;

/// A Python module implemented in Rust.
#[pymodule]
mod env {
    #[pymodule_export]
    use crate::environment::Environment;

    #[pymodule_export]
    use crate::tetris::Snapshot;

    #[pymodule_export]
    use crate::cbor::load_cbor;
}
