# models/

Converted safetensors weights land here. The `*.manifest.json` files are
committed, and so are the **redistributable** weights: the MIT-lineage
detector and the Apache-2.0 foamliu embedder (license notices in
`LICENSES.md`). Everything else is git-ignored — produce it with
`python3 tools/fetch_and_convert.py` (see `tools/README.md`). SHA-256 pins
for the downloaded originals are enforced by the script, and each manifest
records source URL, pin, license notes, exact preprocessing, and
architecture hyperparameters.

Provenance and licensing per docs/adrs/0003 (+ addendum):

| File | Source | License |
|---|---|---|
| `detector-slim320.safetensors` | Faceplugin SDK `version-slim-320.pth` (originally Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | MIT — **committed + shipped** |
| `embedder-foamliu.safetensors` | foamliu/MobileFaceNet release asset `v1.0/mobilefacenet.pt` (inverted-residual MobileFaceNet 112×112 RGB, 128-d, MS-Celeb-1M) | Apache-2.0 (LICENSE at upstream repo root, verified via GitHub API 2026-07-02) — **committed + shipped, the default embedder**; notices in `LICENSES.md` |
| `landmark-mfn68.safetensors` | cunjian/pytorch_face_landmark `checkpoint/mobilefacenet_model_best.pth.tar` (68-pt MobileFaceNet, 112×112) | no LICENSE file upstream; MIT-lineage architecture; fetched at runtime, never redistributed |
| `embedder-mfn.safetensors` | Xiaoccer/MobileFaceNet_Pytorch `model/best/068.ckpt` (MobileFaceNet 112×96 RGB, 128-d, CASIA-WebFace) | no LICENSE file upstream; fetched at runtime, never redistributed (optional alternative — the CLI falls back to it when the foamliu file is absent) |
| `embedder-irn50.safetensors` | optional — converted from user-supplied `irn50_pytorch.npy` via `--irn50` (not published upstream) | n/a |

History note: the foamliu embedder was originally recorded here as
unreachable ("GitHub release downloads are not reachable from the build
environment used to pin these sources; revisit if that changes"). That
changed — the v1.0 release assets are reachable and pinned, and the
"revisit" clause fired on 2026-07-02 (ADR-0003 addendum). The cunjian
landmark repository still publishes no LICENSE (re-checked via the GitHub
licenses API the same day: 404), so the landmark weights remain local-only;
the web demo runs live detection out of the box and collects that one file
via a drop-zone to unlock landmarks/pose/alignment/compare.
