#![recursion_limit = "512"]
mod dqn;
mod environment;
mod memory;
/// Inputs to model:
/// Board: bottom 24 rows with mino (0 = empty)
/// Current piece: Same as board
/// Ghost: Same as board
/// Hold: one number: mino
/// Next queue: five minos
///
/// Outputs from model:
/// Left/Right/None softmax
/// Cw/Ccw/None softmax
/// Hard Drop/Soft Drop/Hold/None softmax
///
/// Things to take into account for evaluating
/// - Holes
/// - Can place the next minos safely
/// - Score
/// - Invalid moves (Hold, Invalid rotation)
/// - Gameover
/// - Max + 2nd Height
/// - Avg Height
/// - Just the possibility of a tetris
///
mod model;
mod ppo;
mod tetris;
mod tui;

use crate::{
    dqn::{DQNConfig, RandomActionThresholdConfig, train_loop},
    model::Model,
};
use burn::{
    backend::Autodiff,
    module::Module,
    record::{DefaultFileRecorder, FullPrecisionSettings},
};

fn main() -> Result<(), burn::record::RecorderError> {
    type Backend = Autodiff<burn::backend::Wgpu>;

    let device = Default::default();

    let model: Model<Backend> = train_loop(
        100_000,
        500,
        &device,
        DQNConfig::new(0.999, 0.05, 0.001, 16),
        RandomActionThresholdConfig::new(0.9, 0.05, 150.),
    );
    model.save_file(
        "./model_saved",
        &DefaultFileRecorder::<FullPrecisionSettings>::default(),
    )
}
