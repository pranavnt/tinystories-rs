use burn::optim::decay::WeightDecayConfig;
use tinystories_rs::{training::ExperimentConfig, data::TinystoriesDataset};
use tinystories_rs::training;

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(512, 1024, 16, 8)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    training::train::<Backend, TinystoriesDataset>(
        if cfg!(target_os = "macos") {
            burn::tensor::Device::<Backend>::Mps
        } else {
            burn::tensor::Device::<Backend>::Cuda(0)
        },
        TinystoriesDataset::train(),
        TinystoriesDataset::test(),
        config,
        "/tmp/text-generation",
    );
}