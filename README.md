# rvFACE

**Rust + WebAssembly face recognition** — a complete port of the
[Faceplugin Open-Source-Face-Recognition-SDK](https://github.com/Faceplugin-ltd/Open-Source-Face-Recognition-SDK)
(Python/PyTorch) to Rust, running natively and in the browser on **WebGPU or CPU**, with a web UI.

![rvFACE web UI demo](docs/media/demo.gif)

| Analyze: detection · 68 landmarks · pose | 1:1 compare: score gauge · threshold-75 verdict |
|---|---|
| ![Analyze pane](docs/media/screenshot-analyze.png) | ![Compare pane](docs/media/screenshot-compare.png) |

## Pipeline

```
image ─► slim-320 SSD detector ─► 68-pt MobileFaceNet landmarks ─► head pose
                                        │
                                        ▼
                          eyes-level alignment (128×128)
                                        │
                                        ▼
                       embedding CNN ─► L2-normalized feature
                                        │
                                        ▼
                     similarity = (dot + 1) × 50   (match > 75)
```

## Workspace

| Path | What |
|---|---|
| `crates/rvface-core` | Framework-free pipeline math (priors, NMS, alignment, pose, similarity, image ops) |
| `crates/rvface-models` | [Burn](https://burn.dev) ports of the three CNNs (CPU: ndarray · WebGPU: wgpu) |
| `crates/rvface-cli` | Native CLI (`rvface detect`, `rvface compare`) |
| `crates/rvface-wasm` | Browser bindings (wasm-bindgen) |
| `web/` | Web UI (Vite + TS): upload/webcam, overlays, 1:1 compare, backend toggle |
| `tools/` | Python: weight conversion → safetensors, golden parity fixtures |
| `docs/adrs/` | Architecture decision records (start at [0001](docs/adrs/0001-project-scope-and-layout.md)) |

## Quick start

```bash
# native
cd rvface
python3 tools/fetch_and_convert.py          # download + convert weights → models/
cargo run -p rvface-cli --release -- compare a.jpg b.png

# browser
cd web && npm install && npm run dev        # weights served from web/public/models/
```

## Status

**Complete.** All three networks ported to Burn with PyTorch golden-parity
green (max|Δ| ~1e-7 on real weights), the full pipeline reproduces the
upstream demo verdict on its own test images (score 78.2 → *same person*),
and the browser runs the identical engine (wasm, 1.42 MB gzipped, CPU with
SIMD128 or WebGPU with automatic CPU fallback) — no mocks anywhere.

- 66 Rust tests: unit math, seven PyTorch parity fixtures, end-to-end on the
  upstream test images ([validation strategy](docs/adrs/0006-testing-validation-optimization.md))
- [Benchmarks](docs/BENCHMARKS.md): native analyze 176 ms, browser ~0.5 s
  (CPU; includes the 5× denormal-weight fix)
- See [ADR-0003](docs/adrs/0003-models-weights-licensing.md) (+ addendum)
  for the weight licensing story. Two weight files are properly licensed and
  **ship with the repo + demo**: the detector (MIT lineage) and the default
  embedder (foamliu/MobileFaceNet, Apache-2.0 — notices in
  [`models/LICENSES.md`](models/LICENSES.md)), so the web demo runs **live
  face detection out of the box**. The landmark checkpoint has **no upstream
  LICENSE file** ([`models/README.md`](models/README.md)) and is never
  redistributed: the tooling fetches it locally and the web demo collects
  that one file via a drop-zone to unlock landmarks/pose/compare. See also
  how to drop in the exact upstream IRN-50 embedder via `--irn50`.

## License & responsible use

Code is [MIT](LICENSE). Only properly licensed model weights are
redistributed — the MIT-lineage detector and the Apache-2.0 foamliu embedder,
with notices in [`models/LICENSES.md`](models/LICENSES.md). The fetch tooling
downloads the remaining third-party checkpoints locally (SHA-256-pinned);
their licensing is documented per-file in
[`models/README.md`](models/README.md) and
[ADR-0003](docs/adrs/0003-models-weights-licensing.md) — review it before any
commercial use.

This is face-recognition software, i.e. biometric processing. It runs entirely
locally (no telemetry, no network calls at inference). It is intended for
consent-based applications — authentication, personal photo tooling, research.
Do **not** use it for surveillance, tracking, or identification of people who
have not consented, and check the biometric-data laws that apply in your
jurisdiction (e.g. GDPR Art. 9, BIPA) before deployment.
