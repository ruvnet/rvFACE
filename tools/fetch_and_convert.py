#!/usr/bin/env python3
"""Download the published weights and convert them to safetensors in ../models/.

Produced files (tensor keys are the ORIGINAL PyTorch state_dict keys, with any
DataParallel ``module.`` prefix stripped — see naming.md):

- models/detector-slim320.safetensors  + .manifest.json
    upstream ``version-slim-320.pth`` (Ultra-Light-Fast-Generic-Face-Detector
    slim-320, MIT lineage)
- models/landmark-pipnet.safetensors   + .manifest.json
    xlite-dev/torchlm PIPNet ResNet-18 68-point landmark net (MIT), released
    asset ``pipnet_resnet18_10x68x32x256_300w.pth`` (256x256 RGB, ImageNet-
    normalized); arch reconstructed in tools/_pipnet_ref.py
- models/embedder-foamliu.safetensors  + .manifest.json
    foamliu/MobileFaceNet release asset ``mobilefacenet.pt`` (Apache-2.0,
    MobileNetV2-style inverted-residual MobileFaceNet, 112x112 RGB, 128-d) —
    the DEFAULT, redistributable embedder (committed + shipped, ADR-0003)
- models/embedder-mfn.safetensors      + .manifest.json
    Xiaoccer/MobileFaceNet_Pytorch ``model/best/068.ckpt`` ArcFace-style
    embedding MobileFaceNet (112x96 RGB, 128-d) — optional alternative,
    no upstream LICENSE, never redistributed
- models/embedder-irn50.safetensors    + .manifest.json   (only with --irn50)
    converted from a user-supplied ``irn50_pytorch.npy`` (upstream never
    published it)

Every download is SHA-256-pinned; a mismatch aborts. Re-running is a no-op
apart from rewriting the converted outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import common
from common import (
    FILES,
    MODELS_DIR,
    ensure,
    load_py_module,
    save_state_dict_safetensors,
    strip_module_prefix,
    write_json,
)


def _load_state_dict(path) -> dict:
    """Load a ``.pth``/``.pt`` file and return its raw state_dict.

    Release assets are pinned by SHA-256 (``ensure``), so ``weights_only=False``
    is safe here; the file may be a bare state_dict, a ``{'state_dict': ...}``
    wrapper, or a pickled ``nn.Module``.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return obj["state_dict"] if "state_dict" in obj else obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def _manifest(name: str, source_key: str, license_: str,
              input_desc: dict, output_desc: str, arch: dict, tensors: dict) -> dict:
    _, url, pin = FILES[source_key]
    return {
        "name": name,
        "file": f"{name}.safetensors",
        "source_url": url,
        "source_sha256": pin,
        "license": license_,
        "input": input_desc,
        "output": output_desc,
        "arch": arch,
        "tensors": tensors,
    }


def convert_detector(paths: dict) -> None:
    sd = torch.load(paths["detector.pth"], map_location="cpu", weights_only=True)
    sd = strip_module_prefix(sd)
    out = MODELS_DIR / "detector-slim320.safetensors"
    tensors = save_state_dict_safetensors(sd, out)
    manifest = _manifest(
        "detector-slim320", "detector.pth",
        "MIT (weights originate from Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB, "
        "MIT; republished unmodified by Faceplugin-ltd/Open-Source-Face-Recognition-SDK, "
        "which itself carries no LICENSE file)",
        {
            "width": 320, "height": 240, "channels": 3, "colorspace": "rgb",
            "mean": [127.0, 127.0, 127.0], "scale": 1.0 / 128.0,
            "layout": "nchw",
        },
        "tuple (confidences [1,4420,2] softmaxed over classes {background, face}, "
        "boxes [1,4420,4] corner-form x1,y1,x2,y2 in normalized image units, not "
        "clipped to [0,1]); this is the is_test=True graph, i.e. before score "
        "filtering and NMS",
        {
            "family": "ultraface-ssd",
            "variant": "slim320",
            "num_classes": 2,
            "num_priors": 4420,
            "base_channel": 16,
            "feature_maps": [[40, 20, 10, 5], [30, 15, 8, 4]],
            "min_boxes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
            "center_variance": 0.1,
            "size_variance": 0.2,
            "source_layer_indexes": [8, 11, 13],
        },
        tensors,
    )
    write_json(MODELS_DIR / "detector-slim320.manifest.json", manifest)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(tensors)} tensors)")


def convert_landmark(paths: dict) -> None:
    import _pipnet_ref

    sd = strip_module_prefix(_load_state_dict(paths["landmark.pth"]))
    # validate against the reconstructed PIPNet ResNet-18 architecture (keys must
    # match the released state_dict exactly) before writing.
    net = _pipnet_ref.build_pipnet()
    net.load_state_dict(sd, strict=True)
    out = MODELS_DIR / "landmark-pipnet.safetensors"
    tensors = save_state_dict_safetensors(sd, out)
    manifest = _manifest(
        "landmark-pipnet", "landmark.pth",
        "MIT (xlite-dev/torchlm; PIPNet ResNet-18). Openly-licensed, redistributable "
        "replacement for the unlicensed cunjian landmark net (ADR-0003); weights "
        "converted verbatim from the released PyTorch state_dict (any DataParallel "
        "'module.' prefix stripped).",
        {
            "width": 256, "height": 256, "channels": 3, "colorspace": "rgb",
            "mean": [0.485, 0.456, 0.406], "scale": 1.0,
            "layout": "nchw",
            "note": "256x256 RGB face crop (detector box expanded 1.2x to a square, "
                    "resized to 256x256), ImageNet-normalized: scale pixels to [0,1] "
                    "(divide by 255), then subtract per-channel mean [0.485,0.456,0.406] "
                    "and divide by per-channel std [0.229,0.224,0.225]. The 'mean' field "
                    "records the ImageNet mean; the matching std division "
                    "[0.229,0.224,0.225] is part of the same transform, applied after "
                    "mean subtraction.",
        },
        "five raw ResNet-18 head score maps (no sigmoid): cls [1,68,8,8], "
        "x [1,68,8,8], y [1,68,8,8], nb_x [1,680,8,8], nb_y [1,680,8,8]; the Rust port "
        "applies the PIPNet heatmap-argmax + x/y offset + neighbor-regression (NRM) "
        "decode to recover the 68 landmark coordinates",
        {
            "family": "pipnet-resnet18",
            "num_lms": 68,
            "num_nb": 10,
            "input_size": [256, 256],
            "net_stride": 32,
        },
        tensors,
    )
    write_json(MODELS_DIR / "landmark-pipnet.manifest.json", manifest)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(tensors)} tensors)")


def convert_embedder(paths: dict) -> None:
    ck = torch.load(paths["embedder.ckpt"], map_location="cpu", weights_only=True)
    sd = strip_module_prefix(ck["net_state_dict"])
    xm = load_py_module(paths["xiaoccer_model.py"], "rvface_xiaoccer_mfn")
    net = xm.MobileFacenet()
    net.load_state_dict(sd, strict=True)
    out = MODELS_DIR / "embedder-mfn.safetensors"
    tensors = save_state_dict_safetensors(sd, out)
    manifest = _manifest(
        "embedder-mfn", "embedder.ckpt",
        "no LICENSE file in Xiaoccer/MobileFaceNet_Pytorch; trained on CASIA-WebFace "
        "(research use); weights are fetched at runtime and never redistributed with "
        "rvFACE (ADR-0003). The properly-licensed Apache-2.0 alternative "
        "(foamliu/MobileFaceNet release asset) is not reachable from this "
        "environment (github.com release downloads blocked); revisit if egress opens.",
        {
            "width": 96, "height": 112, "channels": 3, "colorspace": "rgb",
            "mean": [127.5, 127.5, 127.5], "scale": 1.0 / 128.0,
            "layout": "nchw",
            "note": "SphereFace-style aligned 112x96 face crop; reference eval "
                    "(lfw_eval.py) averages features of the crop and its horizontal "
                    "flip, rvFACE uses the plain crop",
        },
        "embedding [1,128], NOT L2-normalized (normalize downstream before the "
        "upstream similarity formula score=(dot+1)*50)",
        {
            "family": "mobilefacenet",
            "style": "bottleneck",
            "in_channels": 3,
            "input_size": [112, 96],
            "activation": "prelu",
            "conv1": {"out": 64, "kernel": 3, "stride": 2, "pad": 1},
            "dw_conv1": {"out": 64, "kernel": 3, "stride": 1, "pad": 1},
            "bottleneck_setting": [[2, 64, 5, 2], [4, 128, 1, 2], [2, 128, 6, 1],
                                    [4, 128, 1, 2], [2, 128, 2, 1]],
            "bottleneck_setting_columns": ["expansion", "channels", "num_blocks", "first_stride"],
            "conv2": {"out": 512, "kernel": 1},
            "gdc_kernel": [7, 6],
            "gdc_linear_bias": False,
            "embedding_size": 128,
            "output_dim": 128,
        },
        tensors,
    )
    write_json(MODELS_DIR / "embedder-mfn.manifest.json", manifest)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(tensors)} tensors)")


def convert_embedder_foamliu(paths: dict) -> None:
    sd = torch.load(paths["foamliu_embedder.pt"], map_location="cpu", weights_only=True)
    sd = strip_module_prefix(sd)
    # validate against the upstream Apache-2.0 architecture before writing
    fm = common.load_foamliu_mfn(paths["foamliu_mobilefacenet.py"])
    net = fm.MobileFaceNet()
    net.load_state_dict(sd, strict=True)
    out = MODELS_DIR / "embedder-foamliu.safetensors"
    tensors = save_state_dict_safetensors(sd, out)
    manifest = _manifest(
        "embedder-foamliu", "foamliu_embedder.pt",
        "Apache License 2.0 (foamliu/MobileFaceNet publishes a LICENSE file at the repo "
        "root; verified via the GitHub licenses API, spdx_id Apache-2.0, 2026-07-02). "
        "Redistributable: converted weights are committed to this repository and shipped "
        "with the Pages demo, with attribution and the full license text in "
        "models/LICENSES.md. Source: release asset v1.0 mobilefacenet.pt (raw state "
        "dict); trained on MS-Celeb-1M (research dataset).",
        {
            "width": 112, "height": 112, "channels": 3, "colorspace": "rgb",
            "mean": [123.675, 116.28, 103.53], "scale": 1.0 / 255.0,
            "std": [0.229, 0.224, 0.225],
            "layout": "nchw",
            "note": "torchvision ToTensor + Normalize(mean=[0.485,0.456,0.406], "
                    "std=[0.229,0.224,0.225]), folded to pixel domain: "
                    "out[c] = ((pixel - mean[c]) * scale) / std[c]. Reference training "
                    "crops are InsightFace-aligned 112x112; rvFACE bilinear-resizes its "
                    "aligned 128x128 eyes-level crop to 112x112 (documented adaptation, "
                    "same spirit as embedder-mfn)",
        },
        "embedding [1,128], NOT L2-normalized (normalize downstream before the "
        "upstream similarity formula score=(dot+1)*50)",
        {
            "family": "mobilefacenet",
            "style": "inverted-residual-v2",
            "in_channels": 3,
            "input_size": [112, 112],
            "activation": "relu6",
            "inverted_residual_setting": [[2, 64, 5, 2], [4, 128, 1, 2], [2, 128, 6, 1],
                                           [4, 128, 1, 2], [2, 128, 2, 1]],
            "inverted_residual_setting_columns": ["expansion", "channels", "num_blocks",
                                                   "first_stride"],
            "gdc_kernel": [7, 7],
            "embedding_size": 128,
            "output_dim": 128,
        },
        tensors,
    )
    write_json(MODELS_DIR / "embedder-foamliu.manifest.json", manifest)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(tensors)} tensors)")


def convert_irn50(npy_path: Path, paths: dict) -> None:
    """Convert a user-supplied irn50_pytorch.npy via the upstream loader itself.

    ``face_feature/irn50_pytorch.py`` builds the module *from* the npy dict
    ({layer: {'weights','bias','mean','var','scale'}}), so instantiating it with
    the real file performs the authoritative npy-key -> state_dict-key mapping
    (conv 'weights'->'<name>.weight'; bn 'scale'/'bias'/'mean'/'var' ->
    '<name>.weight'/'.bias'/'.running_mean'/'.running_var'; dense
    'weights'/'bias' -> 'fc1_1.weight'/'.bias').
    """
    common.install_import_stubs()
    common.add_sdk_to_path()
    from face_feature.irn50_pytorch import irn50_pytorch

    net = irn50_pytorch(str(npy_path))
    out = MODELS_DIR / "embedder-irn50.safetensors"
    tensors = save_state_dict_safetensors(net.state_dict(), out)
    manifest = {
        "name": "embedder-irn50",
        "file": "embedder-irn50.safetensors",
        "source_url": None,
        "source_path": str(npy_path),
        "source_sha256": common.sha256_path(npy_path),
        "license": "unpublished upstream weights supplied by the user; not redistributable",
        "input": {
            "width": 128, "height": 128, "channels": 3, "colorspace": "bgr",
            "mean": [0.0, 0.0, 0.0], "scale": 1.0 / 256.0,
            "layout": "nchw",
            "note": "aligned 128x128 crop; the 1/256 (not 1/255) divisor is an "
                    "upstream quirk kept deliberately (ADR-0004)",
        },
        "output": "embedding [1,256] = element-wise max of the two 256-halves of the "
                  "512-d fc1_1+bn_fc1 output (maxout); NOT L2-normalized",
        "arch": {
            "family": "irn50",
            "in_channels": 3,
            "input_size": [128, 128],
            "activation": "relu",
            "bn_eps": 9.999999747378752e-06,
            "fc_dim": 512,
            "output_dim": 256,
            "maxout": True,
        },
        "tensors": tensors,
    }
    write_json(MODELS_DIR / "embedder-irn50.manifest.json", manifest)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(tensors)} tensors)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--irn50", type=Path, metavar="PATH",
                    help="optional path to upstream's unpublished irn50_pytorch.npy; "
                         "converts it to models/embedder-irn50.safetensors")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure("detector.pth", "landmark.pth", "embedder.ckpt",
                   "xiaoccer_model.py",
                   "foamliu_embedder.pt", "foamliu_mobilefacenet.py",
                   "test_1.jpg", "test_2.png")
    if args.irn50:
        # irn50 conversion additionally needs the upstream module source
        paths.update(ensure("sdk/irn50_pytorch.py"))

    convert_detector(paths)
    convert_landmark(paths)
    convert_embedder(paths)
    convert_embedder_foamliu(paths)
    if args.irn50:
        if not args.irn50.exists():
            sys.exit(f"ERROR: --irn50 file not found: {args.irn50}")
        convert_irn50(args.irn50, paths)
    print("done")


if __name__ == "__main__":
    main()
