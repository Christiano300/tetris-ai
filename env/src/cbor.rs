use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
use serde::Deserialize;
use tetris_core::tetris::{Action as TetrisAction, Board, Mino, Tetrimino};

use crate::tetris::raw_to_state;

#[allow(dead_code)]
#[derive(Deserialize)]
struct ExportFrame {
    pub board: Board,
    pub piece: Tetrimino,
    pub ghost: Option<Tetrimino>,
    pub hold: Option<Tetrimino>,
    pub next_queue: Vec<Mino>,
    pub score: u32,
    pub level: u8,
    pub inputs: Vec<TetrisAction>,
}

#[pyfunction]
pub fn load_cbor<'py>(bin: Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyList>> {
    let py = bin.py();

    let mut frames: Vec<ExportFrame> = serde_cbor::from_slice(bin.as_bytes())
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to parse CBOR data: {}", e)))?;

    frames.truncate(frames.len() - 250);

    let list = PyList::empty(py);
    for frame in frames {
        let Some(ghost) = frame.ghost else {
            continue;
        };
        if frame.inputs.is_empty() {
            continue;
        }
        let movement = frame
            .inputs
            .iter()
            .find_map(|a| match a {
                TetrisAction::Left => Some(0),
                TetrisAction::Right => Some(1),
                _ => None,
            })
            .unwrap_or(2);
        let rotation = frame
            .inputs
            .iter()
            .find_map(|a| match a {
                TetrisAction::Cw => Some(0),
                TetrisAction::Ccw => Some(1),
                _ => None,
            })
            .unwrap_or(2);
        let action = frame
            .inputs
            .iter()
            .find_map(|a| match a {
                TetrisAction::HardDrop => Some(0),
                TetrisAction::SoftDrop => Some(1),
                TetrisAction::Hold => Some(2),
                _ => None,
            })
            .unwrap_or(3);
        let dict = PyDict::new(py);
        dict.set_item(
            "state",
            raw_to_state(
                &frame.board,
                &frame.piece,
                &ghost,
                frame.hold.as_ref(),
                &frame.next_queue,
            ),
        )?;
        dict.set_item("movement", movement)?;
        dict.set_item("rotation", rotation)?;
        dict.set_item("additional", action)?;

        list.append(dict)?;
    }
    Ok(list)
}
