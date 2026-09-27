use std::{any::Any, collections::HashMap};

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    nn::{
        Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

pub const BOARD_HEIGHT: usize = 24;
pub const BOARD_WIDTH: usize = 10;
pub const BOARD_CHANNELS: usize = 3;

const OUTPUT_MOVE: usize = 3;
const OUTPUT_ROTATE: usize = 3;
const OUTPUT_ACTION: usize = 4;
const OUTPUT_TOTAL: usize = OUTPUT_MOVE + OUTPUT_ROTATE + OUTPUT_ACTION;

const QUEUE_SIZE: usize = 5;
pub const PIECE_INPUTS: usize = QUEUE_SIZE + 1;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.1")]
    dropout: f64,

    #[config(default = "[32, 16]")]
    hidden_size: [usize; 2],
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([3, 6], [3, 3]) // 24 * 10 -> 22 * 8
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            conv2: Conv2dConfig::new([6, 8], [3, 3]) // 22 * 8 -> 20 * 6
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(), // 20 * 6 -> 10 * 3
            conv3: Conv2dConfig::new([8, 12], [3, 3]) // 10 * 3 -> 8 * 1
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            // channels * width * height + hold + queue
            linear1: LinearConfig::new(12 * 8 * 1 + PIECE_INPUTS, self.hidden_size[0]).init(device),
            linear2: LinearConfig::new(self.hidden_size[0], self.hidden_size[1]).init(device),
            // move + rotation + drop/hold
            linear3: LinearConfig::new(self.hidden_size[1], OUTPUT_TOTAL).init(device),
            activation: LeakyReluConfig::new().with_negative_slope(0.1).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool1: MaxPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: LeakyRelu,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    /// # Shapes
    /// - Board: \[3, 24, 10]
    /// - Pieces: \[1 + 5]
    /// - Output: \[3 + 3 + 4]
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut parts = state.split_with_sizes(
            vec![BOARD_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH, PIECE_INPUTS],
            1,
        );
        let mut x = parts.remove(0).reshape([
            -1,
            BOARD_CHANNELS as i32,
            BOARD_HEIGHT as i32,
            BOARD_WIDTH as i32,
        ]);

        x = self.conv1.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);
        x = self.conv2.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);
        x = self.pool1.forward(x);
        x = self.conv3.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);

        let mut x = x.reshape([-1, 12 * 8 * 1]);
        x = Tensor::cat(vec![x, parts.remove(0)], 1);
        x = self.linear1.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);
        x = self.linear2.forward(x);
        x = self.activation.forward(x);
        x = self.linear3.forward(x);
        x = self.dropout.forward(x);
        x = self.activation.forward(x);

        x
    }

    pub fn soft_update(self, policy_net: &Model<B>, tau: f32) -> Self {
        let mut updated_params = HashMap::new();

        // Collect parameters from the policy network
        policy_net.visit(&mut |param: &Param<Tensor<B, _>>| {
            updated_params.insert(param.id.clone(), param.val().clone());
        });

        // Update parameters in the current network
        self.map(&mut |param: Param<Tensor<B, _>>| {
            if let Some(policy_val) = updated_params.get(&param.id) {
                let new_val = param.val() * (1.0 - tau) + policy_val * tau;
                Param::initialized(param.id.clone(), new_val)
            } else {
                param
            }
        })
    }
}
