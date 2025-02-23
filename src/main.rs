use burn::tensor::{backend::ndarray::NdArrayBackend, Tensor};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{Adam, Optimizer};
use burn::train::LearnerBuilder;
use rand::Rng;
use textplots::{Chart, Plot, Shape};

const NUM_SAMPLES: usize = 100;
const NOISE_LEVEL: f64 = 0.5;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 1000;

// Linear Regression Model
#[derive(Module, Debug)]
struct LinearRegression {
    linear: Param<Linear<NdArrayBackend<f64>>>,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            linear: Param::new(Linear::new(&LinearConfig::new(1, 1))),
        }
    }

    fn forward(&self, x: Tensor<NdArrayBackend<f64>, 2>) -> Tensor<NdArrayBackend<f64>, 2> {
        self.linear.forward(x)
    }
}

fn main() {
    // Step 1: Generate Synthetic Data
    let (x_train, y_train) = generate_data(NUM_SAMPLES);
    let (x_test, y_test) = generate_data(20); // For evaluation

    // Convert data to tensors
    let x_train_tensor = Tensor::from_data(x_train);
    let y_train_tensor = Tensor::from_data(y_train);
    let x_test_tensor = Tensor::from_data(x_test);
    let y_test_tensor = Tensor::from_data(y_test);

    // Step 2: Define the Model
    let model = LinearRegression::new();

    // Step 3: Train the Model
    let optimizer = Adam::new(LEARNING_RATE);
    let learner = LearnerBuilder::new()
        .model(model)
        .loss_fn(burn::loss::mean_squared_error())
        .optimizer(optimizer)
        .num_epochs(EPOCHS)
        .build();

    let trained_model = learner.train(x_train_tensor.clone(), y_train_tensor.clone());

    // Step 4: Evaluate the Model
    let predictions = trained_model.forward(x_test_tensor.clone());
    let mse = burn::loss::mean_squared_error().forward(predictions.clone(), y_test_tensor.clone());

    println!("Mean Squared Error on Test Data: {:?}", mse.to_data());

    // Step 5: Plot the Results
    plot_results(&x_test_tensor.to_data(), &y_test_tensor.to_data(), &predictions.to_data());
}

// Function to generate synthetic data
fn generate_data(num_samples: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..num_samples {
        let x: f64 = rng.gen_range(0.0..10.0);
        let noise: f64 = rng.gen_range(-NOISE_LEVEL..NOISE_LEVEL);
        let y = 2.0 * x + 1.0 + noise;

        x_data.push(vec![x]);
        y_data.push(vec![y]);
    }

    (x_data, y_data)
}

// Function to plot results
fn plot_results(x: &[Vec<f64>], y_true: &[Vec<f64>], y_pred: &[Vec<f64>]) {
    let true_points: Vec<(f32, f32)> = x
        .iter()
        .zip(y_true.iter())
        .map(|(xi, yi)| (xi[0] as f32, yi[0] as f32))
        .collect();

    let pred_points: Vec<(f32, f32)> = x
        .iter()
        .zip(y_pred.iter())
        .map(|(xi, yi)| (xi[0] as f32, yi[0] as f32))
        .collect();

    println!("True vs Predicted Data:");
    Chart::new(180, 60, 0.0, 10.0)
        .lineplot(&Shape::Points(&true_points))
        .lineplot(&Shape::Points(&pred_points))
        .display();
}
