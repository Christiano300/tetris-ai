use pyo3::prelude::*;
use tetris_core::tetris::{Game, GameConfig, Phase};

use crate::tetris::{Action, OutputVec, Snapshot, StateVec, game_to_state, step_inner};

/// A simple wrapper around a tetris game
#[pyclass]
pub struct Environment {
    pub game: Game,
    steps: u32,
}

#[pymethods]
impl Environment {
    #[new]
    pub fn new() -> Self {
        Self {
            game: Environment::get_init_game(),
            steps: 0,
        }
    }

    pub fn reset(&mut self) {
        self.game = Environment::get_init_game();
        self.steps = 0;
    }

    pub fn state(&self) -> StateVec {
        game_to_state(&self.game)
    }

    #[pyo3(signature = (action))]
    pub fn step(&mut self, action: OutputVec) -> Snapshot {
        let (reward, done) = step_inner(
            &mut self.game,
            &Action::from_ontput_argmax(action),
            self.steps,
        );
        self.steps += 1;
        Snapshot {
            state: game_to_state(&self.game),
            reward,
            done,
        }
    }
}
impl Environment {
    fn get_init_game() -> Game {
        let mut game = Game::new(GameConfig::default());
        game.level = 11;
        game.phase = Phase::Falling { timer: 0 };
        game
    }
}
