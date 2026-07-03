//! Tier-3 end-to-end tests (ADR-0006): the upstream demo images through the
//! full Rust pipeline with the real converted weights in `models/`.
//!
//! Native-only (needs `std::fs`); skips gracefully with an explanation on
//! stderr when the weights or the cached test images are absent (they are
//! fetched locally by `tools/fetch_and_convert.py`, never committed), the
//! same pattern as the tier-2 parity tests.

use std::path::{Path, PathBuf};
use std::process::Command;

use burn::backend::NdArray;
use rvface_core::align::{
    compute_alignment, eye_centers, ALIGN_EYE_CENTER_X, ALIGN_EYE_DIST, ALIGN_EYE_Y,
};
use rvface_core::similarity::{is_match, similarity};
use rvface_core::Image;
use rvface_models::embedder::{MfnBottleneckConfig, MfnV2Config};
use rvface_models::pipnet::PipnetConfig;
use rvface_models::weights::{Arch, MfnArch, ModelManifest, WeightsError};
use rvface_models::{Embedder, Face, FacePipeline};

type B = NdArray;

/// The two upstream demo photos (`test/1.jpg`, `test/2.png`), cached by the
/// fetch tooling. Upstream's `run.py` prints "same person" for this pair.
const TEST_IMAGES: [&str; 2] = ["tools/.cache/test_1.jpg", "tools/.cache/test_2.png"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_optional(path: &Path) -> Option<Vec<u8>> {
    match std::fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!("skipping: {} absent", path.display());
            None
        }
    }
}

fn manifest(name: &str) -> Option<ModelManifest> {
    let path = repo_root().join(format!("models/{name}.manifest.json"));
    let json = String::from_utf8(read_optional(&path)?).expect("manifest is utf-8");
    Some(serde_json::from_str(&json).expect("manifest parses"))
}

/// Builds a full pipeline from `models/` with the named embedder
/// (`embedder-mfn` Xiaoccer bottleneck or `embedder-foamliu`
/// inverted-residual-v2), or `None` when any weight/manifest file is absent.
fn load_pipeline_with(embedder_name: &str) -> Option<FacePipeline<B>> {
    let device = Default::default();
    let models = repo_root().join("models");
    let detector = read_optional(&models.join("detector-slim320.safetensors"))?;
    let landmark = read_optional(&models.join("landmark-pipnet.safetensors"))?;
    let embedder = read_optional(&models.join(format!("{embedder_name}.safetensors")))?;

    let landmark_config = match manifest("landmark-pipnet")?.arch {
        Arch::Pipnet(arch) => PipnetConfig::from_arch(&arch),
        other => panic!("unexpected landmark arch: {other:?}"),
    };
    let embedder = match manifest(embedder_name)?.arch {
        Arch::MobileFaceNet(MfnArch::Bottleneck(arch)) => Embedder::mobilefacenet_from_safetensors(
            &embedder,
            MfnBottleneckConfig::from_arch(&arch),
            &device,
        ),
        Arch::MobileFaceNet(MfnArch::InvertedResidualV2(arch)) => {
            Embedder::mobilefacenet_v2_from_safetensors(
                &embedder,
                MfnV2Config::from_arch(&arch),
                &device,
            )
        }
        other => panic!("unexpected embedder arch: {other:?}"),
    }
    .expect("embedder weights load");
    Some(
        FacePipeline::from_safetensors(&detector, &landmark, landmark_config, embedder, &device)
            .expect("pipeline weights load"),
    )
}

fn load_pipeline() -> Option<FacePipeline<B>> {
    load_pipeline_with("embedder-mfn")
}

fn load_image(repo_relative: &str) -> Option<Image> {
    let path = repo_root().join(repo_relative);
    if !path.exists() {
        eprintln!("skipping: {} absent", path.display());
        return None;
    }
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Some(Image::new(img.into_raw(), w, h, 3).expect("RGB8 buffer"))
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Per-image tier-3 property checks (ADR-0006).
fn check_faces(label: &str, faces: &[Face]) {
    assert!(!faces.is_empty(), "{label}: no face detected");
    let face = &faces[0];

    // Primary detection is confident.
    assert!(
        face.detection.score > 0.9,
        "{label}: primary score {} <= 0.9",
        face.detection.score
    );

    // All 68 landmarks lie within a modestly expanded detector box (jawline
    // and hairline points legitimately overshoot the tight box a little).
    let b = face.detection.bbox;
    let (mx, my) = (0.2 * (b.x2 - b.x1), 0.2 * (b.y2 - b.y1));
    for (i, p) in face.landmarks.iter().enumerate() {
        assert!(
            p[0] >= b.x1 - mx && p[0] <= b.x2 + mx && p[1] >= b.y1 - my && p[1] <= b.y2 + my,
            "{label}: landmark {i} at {p:?} outside expanded box {b:?} (margin {mx}x{my})"
        );
    }

    // Pose angles are plausible for a demo portrait.
    for (name, angle) in [
        ("yaw", face.pose.yaw),
        ("pitch", face.pose.pitch),
        ("roll", face.pose.roll),
    ] {
        assert!(angle.abs() < 45.0, "{label}: |{name}| = {angle} >= 45");
    }

    // Alignment property: pushing the landmarks through the computed
    // transform puts the eye centers at (44, 48) / (84, 48) within 1 px.
    let t = compute_alignment(
        &face.landmarks,
        ALIGN_EYE_Y,
        ALIGN_EYE_CENTER_X,
        ALIGN_EYE_DIST,
    )
    .expect("real-face landmarks are non-degenerate");
    let mut transformed = face.landmarks;
    for p in transformed.iter_mut() {
        *p = t.apply(*p);
    }
    let (left, right) = eye_centers(&transformed);
    assert!(
        (left[0] - 44.0).abs() <= 1.0 && (left[1] - 48.0).abs() <= 1.0,
        "{label}: aligned left eye at {left:?}, expected (44, 48) +-1"
    );
    assert!(
        (right[0] - 84.0).abs() <= 1.0 && (right[1] - 48.0).abs() <= 1.0,
        "{label}: aligned right eye at {right:?}, expected (84, 48) +-1"
    );

    // Embeddings come out L2-normalized.
    for (i, f) in faces.iter().enumerate() {
        let norm = l2_norm(&f.embedding);
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "{label}: face {i} embedding norm {norm}"
        );
    }

    // Self-similarity of an embedding with itself is the score ceiling.
    let self_score = similarity(&face.embedding, &face.embedding);
    assert!(
        (self_score - 100.0).abs() <= 0.01,
        "{label}: self-similarity {self_score} != 100 +-0.01"
    );
}

#[test]
fn e2e_pipeline_on_upstream_test_images() {
    let Some(pipeline) = load_pipeline() else {
        return;
    };
    let (Some(img1), Some(img2)) = (load_image(TEST_IMAGES[0]), load_image(TEST_IMAGES[1])) else {
        return;
    };

    let faces1 = pipeline.analyze(&img1, 5).expect("analyze test_1");
    let faces2 = pipeline.analyze(&img2, 5).expect("analyze test_2");
    check_faces("test_1.jpg", &faces1);
    check_faces("test_2.png", &faces2);

    // Cross-image similarity, upstream run.py semantics (features[0] each).
    // The two demo photos are the same person and the upstream demo prints
    // "same person"; the open-licensed PIPNet landmarks (ArcFace-aligned
    // 112x112 crop) with the legacy Xiaoccer MobileFaceNet embedder agree.
    // Measured 82.241 (NdArray f32, 2026-07) — asserted as a range around
    // that value so a legitimate rebuild of the weights doesn't false-alarm,
    // while any preprocessing regression (channel order, crop math, ArcFace
    // alignment, ImageNet normalization, PIPNet decode) still trips it.
    let score = similarity(&faces1[0].embedding, &faces2[0].embedding);
    println!("cross-image similarity (MFN embedder) = {score:.3}");
    assert!(
        is_match(score),
        "cross-image score {score} <= 75: verdict no longer matches upstream's 'same person'"
    );
    assert!(
        (80.0..85.0).contains(&score),
        "cross-image score {score} outside the empirical range [80, 85) (measured 82.241)"
    );
}

/// Same flow with the default, redistributable foamliu embedder
/// (Apache-2.0, `embedder-foamliu.safetensors` — the one the demo ships).
#[test]
fn e2e_pipeline_foamliu_embedder() {
    let Some(pipeline) = load_pipeline_with("embedder-foamliu") else {
        return;
    };
    let (Some(img1), Some(img2)) = (load_image(TEST_IMAGES[0]), load_image(TEST_IMAGES[1])) else {
        return;
    };

    let faces1 = pipeline.analyze(&img1, 5).expect("analyze test_1");
    let faces2 = pipeline.analyze(&img2, 5).expect("analyze test_2");
    check_faces("test_1.jpg (foamliu)", &faces1);
    check_faces("test_2.png (foamliu)", &faces2);

    // Measured 82.771 (NdArray f32, 2026-07) — this is the open-licensed
    // default the demo ships: PIPNet landmarks + ArcFace-template 112x112
    // alignment (the foamliu embedder's training alignment) + foamliu
    // MobileFaceNet-V2. The verdict matches the upstream demo's "same person";
    // asserted as a range around that value so a legitimate weight rebuild
    // doesn't false-alarm while any preprocessing regression (channel order,
    // crop math, ArcFace alignment, ImageNet normalization, PIPNet decode)
    // still trips it.
    let score = similarity(&faces1[0].embedding, &faces2[0].embedding);
    println!("cross-image similarity (foamliu embedder) = {score:.3}");
    assert!(
        is_match(score),
        "cross-image score {score} <= 75: verdict no longer matches upstream's 'same person'"
    );
    assert!(
        (80.0..85.0).contains(&score),
        "cross-image score {score} outside the empirical range [80, 85) (measured 82.771)"
    );

    // Same-person, same-pose control: an image against its own mirror scores
    // far above threshold (measured 99.090) — the embedding is pose-sensitive
    // but identity-consistent.
    let flipped = flip_horizontal(&img1);
    let faces_flip = pipeline.analyze(&flipped, 5).expect("analyze flipped");
    let flip_score = similarity(&faces1[0].embedding, &faces_flip[0].embedding);
    println!("mirror similarity (foamliu embedder) = {flip_score:.3}");
    assert!(
        flip_score > 90.0,
        "mirror-pair score {flip_score} <= 90 (measured 99.090)"
    );
}

/// Horizontal mirror of an RGB8 image (test helper).
fn flip_horizontal(img: &Image) -> Image {
    let (w, h) = (img.width, img.height);
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                data[(y * w + x) * 3 + c] = img.get(w - 1 - x, y, c);
            }
        }
    }
    Image::new(data, w, h, 3).expect("RGB8 buffer")
}

/// Detector-only partial mode (the Pages demo out of the box): `detect`
/// finds the face, `analyze` fails with `MissingStage`.
#[test]
fn e2e_detector_only_partial_mode() {
    let models = repo_root().join("models");
    let Some(detector) = read_optional(&models.join("detector-slim320.safetensors")) else {
        return;
    };
    let Some(img) = load_image(TEST_IMAGES[0]) else {
        return;
    };
    let pipeline =
        FacePipeline::<B>::detector_only_from_safetensors(&detector, &Default::default())
            .expect("detector weights load");
    assert!(!pipeline.is_full());

    let detections = pipeline.detect(&img, 5).expect("detect");
    assert!(!detections.is_empty(), "no face detected in partial mode");
    assert!(detections[0].score > 0.9);

    match pipeline.analyze(&img, 5) {
        Err(WeightsError::MissingStage(stage)) => assert_eq!(stage, "embedder"),
        other => panic!("expected MissingStage, got {other:?}"),
    }
}

/// Drives the actual `rvface` binary end-to-end (ADR-0006: "CLI `rvface
/// compare` is the harness"). The CLI prefers the committed Apache-2.0
/// foamliu embedder; the landmark weights are still local-only.
#[test]
fn e2e_cli_compare_verdict() {
    let root = repo_root();
    let models = root.join("models");
    for file in [
        "detector-slim320.safetensors",
        "landmark-pipnet.safetensors",
        "embedder-foamliu.safetensors",
    ] {
        if !models.join(file).exists() {
            eprintln!("skipping: models/{file} absent");
            return;
        }
    }
    let (img1, img2) = (root.join(TEST_IMAGES[0]), root.join(TEST_IMAGES[1]));
    if !img1.exists() || !img2.exists() {
        eprintln!("skipping: cached test images absent");
        return;
    }

    let output = Command::new(env!("CARGO_BIN_EXE_rvface"))
        .arg("compare")
        .arg(&img1)
        .arg(&img2)
        .arg("--models-dir")
        .arg(&models)
        .output()
        .expect("spawn rvface");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "rvface compare failed: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("score = "), "missing score line: {stdout}");
    assert!(
        stdout.contains("same person"),
        "expected the upstream demo verdict 'same person': {stdout}"
    );
}
