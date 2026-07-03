# models/

Converted safetensors weights land here. The `*.manifest.json` files are
committed, and so are the **redistributable** weights: the MIT-lineage
detector, the MIT PIPNet landmark, and the Apache-2.0 foamliu embedder
(license notices in `LICENSES.md`). Everything else is git-ignored — produce it with
`python3 tools/fetch_and_convert.py` (see `tools/README.md`). SHA-256 pins
for the downloaded originals are enforced by the script, and each manifest
records source URL, pin, license notes, exact preprocessing, and
architecture hyperparameters.

Provenance and licensing per docs/adrs/0003 (+ addendum + update):

| File | Source | License |
|---|---|---|
| `detector-slim320.safetensors` | Faceplugin SDK `version-slim-320.pth` (originally Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | MIT — **committed + shipped** |
| `landmark-pipnet.safetensors` | xlite-dev/torchlm release asset [`pipnet_resnet18_10x68x32x256_300w.pth`](https://github.com/xlite-dev/torchlm/releases/download/torchlm-0.1.6-alpha/pipnet_resnet18_10x68x32x256_300w.pth) (PIPNet ResNet-18, 300W 68-pt; 256×256 RGB, ImageNet-normalized `x/255`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`; SHA-256 `d51c5c5de391770e2ebd491a9af659f32f07e6edc61ec0e5f727e2d1689a781f`) | MIT — **committed + shipped, the default landmark** |
| `embedder-foamliu.safetensors` | foamliu/MobileFaceNet release asset `v1.0/mobilefacenet.pt` (inverted-residual MobileFaceNet, 112×112 **ArcFace-aligned** RGB, 128-d, MS-Celeb-1M) | Apache-2.0 (LICENSE at upstream repo root, verified via GitHub API 2026-07-02) — **committed + shipped, the default embedder**; notices in `LICENSES.md` |
| `embedder-mfn.safetensors` | Xiaoccer/MobileFaceNet_Pytorch `model/best/068.ckpt` (MobileFaceNet 112×96 RGB, 128-d, CASIA-WebFace) | no LICENSE file upstream; optional local-only alternative — no longer a default and no longer fetched |
| `embedder-irn50.safetensors` | optional — converted from user-supplied `irn50_pytorch.npy` via `--irn50` (not published upstream) | n/a |

History note: the foamliu embedder was originally recorded here as
unreachable ("GitHub release downloads are not reachable from the build
environment used to pin these sources; revisit if that changes"). That
changed — the v1.0 release assets are reachable and pinned, and the
"revisit" clause fired on 2026-07-02 (ADR-0003 addendum). The unlicensed
cunjian landmark was then retired in favor of the MIT PIPNet ResNet-18
landmark above (ADR-0003 update), so all three default weights are now
committed + shipped and the web demo runs the **full** pipeline out of the
box — detection, landmarks, pose, ArcFace-template alignment, and compare —
with **no drop-zone**. On the two upstream demo photos the cross-image
"same person" similarity is **82.771** (self-compare 100). The legacy
`landmark-mfn68` (cunjian) and `embedder-mfn` (Xiaoccer) ports keep their
manifests for reference but are no longer defaults and are no longer fetched.
