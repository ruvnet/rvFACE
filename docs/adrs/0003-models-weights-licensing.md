# ADR-0003: Model weights sourcing, conversion, and licensing

- Status: Accepted (amended 2026-07-02 — see Addendum + Update)
- Date: 2026-07-02

## Context

The upstream repository publishes:

| Stage | Architecture code | Pretrained weights |
|---|---|---|
| Detector (slim-320 SSD) | ✅ `face_detect/vision/**` | ✅ `version-slim-320.pth` (1.06 MB) |
| Landmarks (MobileFaceNet-68, 64×64 gray, 136 out) | ✅ `face_landmark/MobileFaceNet.py` | ❌ not published |
| Embedder (IRN-50, 128×128 RGB) | ✅ `face_feature/irn50_pytorch.py` | ❌ `irn50_pytorch.npy` not published |
| Alignment / pose helpers | ❌ `face_util/`, `face_pose/` absent | — |

The upstream repo also has **no LICENSE file** (README badge says
"Open Source"). The detector code and weights originate from
[Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
(MIT). The landmark architecture matches
[cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)
(MIT lineage), which *does* publish a compatible-family 68-point
MobileFaceNet checkpoint (112×112 RGB input variant).

## Decision

1. **Port all three architectures faithfully** in Burn, including IRN-50,
   so the port is complete with respect to everything upstream published.
2. **Architecture correctness is validated without pretrained weights**:
   `tools/` generates *deterministic random* weights per network, runs the
   PyTorch reference implementation, and emits golden input/output pairs;
   the Rust ports must reproduce them (ADR-0006). This decouples "is the
   port correct?" from "are the original weights available?".
3. **Runnable defaults** use openly obtainable weights:
   - Detector: upstream `version-slim-320.pth` (MIT lineage).
   - Landmarks: cunjian's 68-point MobileFaceNet checkpoint; the Rust
     MobileFaceNet is **parameterized** (input channels, spatial size, GDC
     kernel, output dim) so it expresses both the upstream 64×64-gray
     variant and the 112×112-RGB checkpoint variant.
   - Embedder: an openly licensed ArcFace-style MobileFaceNet embedding
     checkpoint (same parameterized architecture, 112×112 RGB); the IRN-50
     port remains available for users who possess `irn50_pytorch.npy`,
     which `tools/convert_weights.py` accepts as an optional input.
4. **Distribution format is safetensors**, produced by `tools/` from the
   original `.pth`/`.pth.tar`/`.npy` files with canonical, stable tensor
   names (documented in `tools/`). Weights are git-ignored and fetched or
   converted locally; the web UI loads them over HTTP at runtime.

## Consequences

- rvFACE works out of the box for detection + landmarks + alignment + pose,
  and for recognition with the substitute embedder; bit-parity with
  upstream's *unpublished* embedder deployment is impossible for anyone
  without `irn50_pytorch.npy`, and this is documented rather than hidden.
- Because the default landmark/embedder weights differ from upstream's
  private ones, the *pipeline semantics* (crop, normalization, similarity
  scale, threshold 75) are still ported exactly (ADR-0004), so scores stay
  in the upstream scale and a user-supplied IRN-50 drops in cleanly.
- No third-party weights enter this repository; licensing notes ship in
  `models/README.md` alongside download provenance and SHA-256 pins.

## Addendum (2026-07-02): the foamliu "revisit" clause fired

`models/README.md` recorded that an Apache-2.0 embedder alternative
(foamliu/MobileFaceNet) existed but its GitHub release assets were not
reachable from the environment that pinned the original sources — "revisit
if that changes". It changed:

- **foamliu/MobileFaceNet is Apache-2.0** (LICENSE file at the repo root;
  verified via the GitHub licenses API, `spdx_id: Apache-2.0`) and its
  v1.0 release assets are reachable. The raw state dict
  (`mobilefacenet.pt`, 4 MB) is fetched SHA-256-pinned and converted to
  `models/embedder-foamliu.safetensors`.
- Consequence: **the "no third-party weights enter this repository" rule is
  relaxed for properly licensed weights.** The converted foamliu embedder
  is committed (in `models/` and `web/public/models/`) and ships with the
  public Pages demo, with attribution + the full Apache-2.0 text in
  `models/LICENSES.md` and the license note in its manifest. It replaces
  the Xiaoccer checkpoint as the **default** embedder (CLI prefers it; the
  web demo fetches only it); the Xiaoccer path remains as a local-only
  alternative.
- The architecture is the same inverted-bottleneck MobileFaceNet family but
  not manifest-expressible on the Xiaoccer port (ReLU6 vs PReLU, MobileNetV2
  state-dict layout, 112×112 input, ImageNet per-channel normalization,
  biased head conv), so it landed as a second manifest-driven variant
  (`style: "inverted-residual-v2"`, `MobileFaceNetV2Embedder`), validated by
  a real-weights golden fixture (`embedder-foamliu-real`).
- The cunjian landmark repository was re-checked the same day and still
  published **no LICENSE** (GitHub licenses API: 404), so at that point the
  landmark weights stayed never-committed/never-deployed: with detector +
  embedder shipped, the web demo ran **live detection out of the box**
  (detector-only partial mode, ADR-0005) while the landmark net alone was
  collected at runtime via a one-file drop-zone. That interim gap is now
  closed — see the Update below.

## Update (2026-07-02): open-licensed PIPNet landmark closes the pipeline

The last non-redistributable default — the unlicensed cunjian landmark — is
**retired** in favor of an MIT checkpoint, so every network the demo runs is now
openly licensed and the pipeline ships **complete, with no runtime drop-zone**:

- **Landmarks — PIPNet ResNet-18 (MIT).** Released asset
  `pipnet_resnet18_10x68x32x256_300w.pth` from
  [xlite-dev/torchlm](https://github.com/xlite-dev/torchlm) (MIT), converted to
  `models/landmark-pipnet.safetensors`. It is a stock torchvision ResNet-18
  stem+trunk with five parallel 1×1-conv heads (`cls`/`x`/`y`/`nb_x`/`nb_y`).
  Input is a **256×256 RGB** crop (detector box expanded 1.2× to a square),
  ImageNet-normalized (`x/255`, mean `[0.485,0.456,0.406]`, std
  `[0.229,0.224,0.225]`). The five raw score maps (`[1,68,8,8]` ×3 +
  `[1,680,8,8]` ×2, net stride 32, 10 neighbors) are decoded in the Rust port
  via PIPNet heatmap-argmax + x/y offset + neighbor-regression (NRM) into the 68
  landmark coordinates — a different decode than the retired cunjian
  MobileFaceNet's direct coordinate regression. SHA-256-pinned
  (`d51c5c5d…a781f`), converted verbatim by `tools/fetch_and_convert.py`, and
  **committed + deployed** to the GitHub Pages demo.
- **Alignment — ArcFace-template 112×112.** The foamliu MobileFaceNet-V2
  embedder (Apache-2.0, unchanged from the Addendum) is now fed a **112×112
  ArcFace-aligned** crop: a 2-point eye-based similarity transform places the
  detected eye centers at the canonical ArcFace 112×112 template positions,
  replacing the previous eyes-level 128×128 crop. The rest of the pipeline
  semantics (ImageNet normalization, L2-normalized 128-d embedding,
  `score = (dot + 1) × 50`, threshold 75) are unchanged (ADR-0004).
- **Result.** With PIPNet landmarks driving the ArcFace-template alignment, the
  cross-image "same person" similarity on the two upstream demo photos rises to
  **82.771** (self-compare 100). The detector (slim-320, MIT) is unchanged, so
  the shipped set is detector (MIT) + PIPNet landmarks (MIT) + foamliu embedder
  (Apache-2.0): the CI Pages job runs `fetch_and_convert.py` before
  `web/build-wasm.sh`, and the hosted demo auto-loads every weight and runs the
  **full** analyze/compare pipeline with **no runtime drop-zone**.

The cunjian 68-pt MobileFaceNet (`landmark-mfn68`) and Xiaoccer MobileFaceNet
(`embedder-mfn`) ports/manifests remain in the tree for reference but are **no
longer defaults** and are no longer fetched. The IRN-50 path still applies: the
exact upstream embedder drops in via `tools/fetch_and_convert.py --irn50
<irn50_pytorch.npy>` for users who possess it.
