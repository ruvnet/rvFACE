//! PIPNet ResNet-18 68-point face landmark network (Burn port).
//!
//! MIT-licensed replacement for the unlicensed cunjian landmark net (ADR-0003).
//! The backbone is a stock torchvision ResNet-18 stem + trunk (`conv1`/`bn1`/
//! `maxpool`, then `layer1..layer4` of two `BasicBlock`s each); five parallel
//! 1x1 conv heads tap the 512x8x8 trunk output and produce the raw PIPNet
//! score maps. State-dict keys are the verbatim torchvision names
//! (`conv1.weight`, `layer2.0.downsample.0.weight`, `cls_layer.weight`, ...).
//!
//! [`PipnetLandmark::forward`] returns the FIVE raw conv outputs with **no**
//! sigmoid applied (`cls`, `x`, `y`, `nb_x`, `nb_y`); score decoding into
//! pixel landmarks happens downstream.

use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::module::max_pool2d;
use burn::tensor::Tensor;

use crate::ops::{batch_norm2d, conv2d, conv2d_biased, TORCH_BN_EPS};
use crate::weights::{PipnetArch, Weights, WeightsError};

/// Variant parameters of the PIPNet landmark head (decode metadata; the
/// ResNet-18 backbone itself is fixed, so channel widths and conv groups are
/// implied by the stored weight shapes).
#[derive(Debug, Clone)]
pub struct PipnetConfig {
    /// Number of landmarks (68 for the 300W meanface).
    pub num_lms: usize,
    /// Neighbour count per landmark (`nb_x`/`nb_y` have `num_lms * num_nb`
    /// channels).
    pub num_nb: usize,
    /// Input spatial size `[height, width]`.
    pub input_size: [usize; 2],
    /// Backbone total stride (input_size / feature-map size = 32).
    pub net_stride: usize,
}

impl PipnetConfig {
    /// PIPNet ResNet-18 / 300W defaults (68 landmarks, 10 neighbours, 256x256
    /// input, stride 32 → 8x8 score maps).
    pub fn resnet18_300w() -> Self {
        Self {
            num_lms: 68,
            num_nb: 10,
            input_size: [256, 256],
            net_stride: 32,
        }
    }

    /// Builds the config from a manifest `arch` block.
    pub fn from_arch(arch: &PipnetArch) -> Self {
        Self {
            num_lms: arch.num_lms,
            num_nb: arch.num_nb,
            input_size: arch.input_size,
            net_stride: arch.net_stride,
        }
    }
}

/// The five raw PIPNet head outputs (each `[N, C, H, W]`, no activation):
/// classification heat-map plus x/y offset and neighbour x/y offset maps.
pub struct PipnetOutputs<B: Backend> {
    /// `cls_layer`: `[N, num_lms, H, W]` landmark presence score map.
    pub cls: Tensor<B, 4>,
    /// `x_layer`: `[N, num_lms, H, W]` in-cell x offset map.
    pub x: Tensor<B, 4>,
    /// `y_layer`: `[N, num_lms, H, W]` in-cell y offset map.
    pub y: Tensor<B, 4>,
    /// `nb_x_layer`: `[N, num_lms * num_nb, H, W]` neighbour x offset map.
    pub nb_x: Tensor<B, 4>,
    /// `nb_y_layer`: `[N, num_lms * num_nb, H, W]` neighbour y offset map.
    pub nb_y: Tensor<B, 4>,
}

/// PIPNet ResNet-18 68-point landmark network (`torchlm` PIPNet, MIT).
pub struct PipnetLandmark<B: Backend> {
    weights: Weights<B>,
    #[allow(dead_code)]
    config: PipnetConfig,
}

impl<B: Backend> PipnetLandmark<B> {
    /// Wraps a loaded weight store with the variant config (canonical
    /// torchvision ResNet-18 keys + `{cls,x,y,nb_x,nb_y}_layer.{weight,bias}`).
    pub fn new(weights: Weights<B>, config: PipnetConfig) -> Self {
        Self { weights, config }
    }

    /// torchvision ResNet `BasicBlock` `layer{prefix}`: two 3x3 convs (first
    /// strided) each followed by BN, an optional 1x1 strided `downsample`
    /// projection on the identity path, and a final ReLU on the sum. All
    /// convs are bias-free; `stride` applies to `conv1` and to `downsample`.
    fn basic_block(
        &self,
        x: Tensor<B, 4>,
        prefix: &str,
        stride: usize,
        downsample: bool,
    ) -> Result<Tensor<B, 4>, WeightsError> {
        let w = &self.weights;
        let identity = x.clone();

        let out = conv2d(
            x.clone(),
            w,
            &format!("{prefix}.conv1.weight"),
            None,
            [stride, stride],
            [1, 1],
        )?;
        let out = batch_norm2d(out, w, &format!("{prefix}.bn1"), TORCH_BN_EPS)?;
        let out = relu(out);
        let out = conv2d(out, w, &format!("{prefix}.conv2.weight"), None, [1, 1], [1, 1])?;
        let out = batch_norm2d(out, w, &format!("{prefix}.bn2"), TORCH_BN_EPS)?;

        let identity = if downsample {
            let id = conv2d(
                identity,
                w,
                &format!("{prefix}.downsample.0.weight"),
                None,
                [stride, stride],
                [0, 0],
            )?;
            batch_norm2d(id, w, &format!("{prefix}.downsample.1"), TORCH_BN_EPS)?
        } else {
            identity
        };

        Ok(relu(out + identity))
    }

    /// One ResNet stage: `block0` (given `stride`, with `downsample`) then
    /// `block1` (stride 1, no downsample).
    fn layer(
        &self,
        x: Tensor<B, 4>,
        name: &str,
        stride: usize,
        downsample: bool,
    ) -> Result<Tensor<B, 4>, WeightsError> {
        let x = self.basic_block(x, &format!("{name}.0"), stride, downsample)?;
        self.basic_block(x, &format!("{name}.1"), 1, false)
    }

    /// Runs the network on an ImageNet-normalized `[N, 3, 256, 256]` RGB
    /// tensor. Returns the five raw head score maps (no sigmoid).
    pub fn forward(&self, x: Tensor<B, 4>) -> Result<PipnetOutputs<B>, WeightsError> {
        let w = &self.weights;

        // Stem: Conv(3,64,k7,s2,p3) -> BN -> ReLU -> MaxPool(k3,s2,p1).
        let x = conv2d(x, w, "conv1.weight", None, [2, 2], [3, 3])?;
        let x = batch_norm2d(x, w, "bn1", TORCH_BN_EPS)?;
        let x = relu(x);
        let x = max_pool2d(x, [3, 3], [2, 2], [1, 1], [1, 1]);

        // Trunk: layer1 (s1, no ds) then layer2/3/4 (block0 s2 + downsample).
        let x = self.layer(x, "layer1", 1, false)?;
        let x = self.layer(x, "layer2", 2, true)?;
        let x = self.layer(x, "layer3", 2, true)?;
        let x = self.layer(x, "layer4", 2, true)?;

        // Five parallel 1x1 conv heads (bias=true) on the 512x8x8 trunk output.
        let cls = conv2d_biased(x.clone(), w, "cls_layer", [1, 1], [0, 0])?;
        let ox = conv2d_biased(x.clone(), w, "x_layer", [1, 1], [0, 0])?;
        let oy = conv2d_biased(x.clone(), w, "y_layer", [1, 1], [0, 0])?;
        let nb_x = conv2d_biased(x.clone(), w, "nb_x_layer", [1, 1], [0, 0])?;
        let nb_y = conv2d_biased(x, w, "nb_y_layer", [1, 1], [0, 0])?;

        Ok(PipnetOutputs {
            cls,
            x: ox,
            y: oy,
            nb_x,
            nb_y,
        })
    }
}
