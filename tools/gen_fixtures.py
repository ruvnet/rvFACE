#!/usr/bin/env python3
"""Generate golden parity fixtures for the Rust ports (ADR-0006 tier 2).

Runs the upstream PyTorch reference implementations verbatim (sources are
downloaded into tools/.cache/ — never vendored, ADR-0003) on fixed inputs and
records input/output pairs, plus deterministic random weights where the real
ones are unpublished. The Rust tests load the same weights/inputs and must
match within each fixture's manifest tolerance
(max|delta| <= 1e-4 * max(1, max|output|)).

Fixtures (tools/fixtures/):

  A detector-real     slim-320 SSD, real weights, pre-NMS confidences+boxes
  B detector-rand     same net, seeded random weights (seed 1234)
  C landmark64-rand   upstream 64x64-gray MobileFaceNet-136 (seed 5678)
  D landmark-pipnet-real  PIPNet ResNet-18 (MIT), real weights, seed-7 input
  E irn50-rand        upstream IRN-50 embedder, seeded random weights (9012)
  F embedder-mfn-real Xiaoccer 112x96 MobileFaceNet-128, real checkpoint
  G embedder-foamliu-real  foamliu 112x112 inverted-residual MobileFaceNet-128,
                           real Apache-2.0 release weights

Arrays are float32 .npz; synthetic weights are .safetensors with the original
state_dict keys; each fixture has a small JSON manifest and fixtures/INDEX.json
lists them all. Random weights use recipe "randn-kaiming-v1" (see naming.md and
common.fill_random_state — the plain randn*0.02 recipe was measured to produce
input-INDEPENDENT outputs on these depths and was rejected).
"""

from __future__ import annotations

import numpy as np
import torch

import common
from common import (
    FIXTURES_DIR,
    MODELS_DIR,
    ensure,
    fill_random_state,
    load_py_module,
    pseudo_image,
    save_state_dict_safetensors,
    strip_module_prefix,
    write_json,
)

TOLERANCE = 1e-4
RECIPE = "randn-kaiming-v1"


def _save_fixture(name: str, arrays: dict[str, torch.Tensor], manifest: dict) -> dict:
    data_path = FIXTURES_DIR / f"{name}.npz"
    np.savez_compressed(
        data_path, **{k: v.detach().numpy().astype(np.float32) for k, v in arrays.items()}
    )
    keys = list(arrays.keys())
    # absolute tolerance, scaled to the outputs' magnitude: 1e-4 is the fp32
    # conv-reordering allowance for O(1) values (ADR-0006); random-weight nets
    # can legitimately emit O(100) values, where 1e-4 absolute would demand
    # ~1e-6 relative precision fp32 cannot guarantee across backends.
    out_absmax = max(float(v.abs().max()) for k, v in arrays.items() if k != "input")
    tolerance = TOLERANCE * max(1.0, out_absmax)
    manifest = {
        "fixture": name,
        **manifest,
        "input_file": f"{name}.npz#input",
        "output_files": [f"{name}.npz#{k}" for k in keys if k != "input"],
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtype": "float32",
        "tolerance": tolerance,
        "tolerance_rule": "max|delta| <= 1e-4 * max(1, max|output|), absolute",
    }
    write_json(FIXTURES_DIR / f"{name}.json", manifest)
    print(f"  {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in arrays.items()))
    return {
        "name": name,
        "manifest": f"{name}.json",
        "data": f"{name}.npz",
        **({"weights": manifest["weights"]} if "weights" in manifest else {}),
    }


# --------------------------------------------------------------------------- detector


def make_detector_fixtures(paths) -> list[dict]:
    # mirror upstream detect_imgs.py import order: define_img_size(320) MUST
    # run before importing mb_tiny_fd (it populates fd_config.priors)
    from face_detect.vision.ssd.config.fd_config import define_img_size

    define_img_size(320)
    from face_detect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd

    entries = []
    x = pseudo_image(3407, (1, 3, 240, 320), "det")

    # A: real weights
    net = create_mb_tiny_fd(2, is_test=True, device="cpu")
    sd = torch.load(paths["detector.pth"], map_location="cpu", weights_only=True)
    net.load_state_dict(strip_module_prefix(sd))
    net.eval()
    with torch.no_grad():
        confidences, boxes = net(x)
    entries.append(_save_fixture(
        "detector-real",
        {"input": x, "confidences": confidences, "boxes": boxes},
        {
            "net": "upstream face_detect slim-320 SSD, create_mb_tiny_fd(2, is_test=True), eval mode",
            "weights": "real:models/detector-slim320.safetensors",
            "seed": None,
            "input_seed": 3407,
            "input_domain": "(randint(0,256) - 127) / 128",
            "notes": "confidences softmaxed over {background, face}; boxes decoded to "
                     "corner form in normalized image units (center_variance 0.1, "
                     "size_variance 0.2; priors clipped to [0,1] but decoded boxes may "
                     "spill slightly outside); both recorded BEFORE thresholding/NMS",
        },
    ))

    # B: seeded random weights (same input)
    net = create_mb_tiny_fd(2, is_test=True, device="cpu")
    fill_random_state(net, 1234)
    net.eval()
    wdesc = save_state_dict_safetensors(net.state_dict(), FIXTURES_DIR / "detector-rand.safetensors")
    with torch.no_grad():
        confidences, boxes = net(x)
    entries.append(_save_fixture(
        "detector-rand",
        {"input": x, "confidences": confidences, "boxes": boxes},
        {
            "net": "upstream face_detect slim-320 SSD, create_mb_tiny_fd(2, is_test=True), eval mode",
            "weights": "detector-rand.safetensors",
            "seed": 1234,
            "weight_recipe": RECIPE,
            "input_seed": 3407,
            "input_domain": "(randint(0,256) - 127) / 128",
            "num_weight_tensors": len(wdesc),
        },
    ))
    return entries


# --------------------------------------------------------------------------- landmarks


def make_landmark64_fixture() -> dict:
    # upstream 64x64 grayscale variant, embedding_size=136 (ADR-0003)
    from face_landmark.MobileFaceNet import MobileFaceNet

    net = MobileFaceNet([64, 64], 136)
    fill_random_state(net, 5678)
    net.eval()
    wdesc = save_state_dict_safetensors(net.state_dict(), FIXTURES_DIR / "landmark64-rand.safetensors")
    x = pseudo_image(5679, (1, 1, 64, 64), "unit")
    with torch.no_grad():
        out = net(x)
    return _save_fixture(
        "landmark64-rand",
        {"input": x, "landmarks": out},
        {
            "net": "upstream face_landmark/MobileFaceNet.py MobileFaceNet([64,64], 136): "
                   "1-channel input, ReLU, residual blocks 3/4/2, channels 32/64, "
                   "GDC kernel 4x4, Linear(512,136,bias=True)",
            "weights": "landmark64-rand.safetensors",
            "seed": 5678,
            "weight_recipe": RECIPE,
            "input_seed": 5679,
            "input_domain": "randint(0,256) / 255",
            "num_weight_tensors": len(wdesc),
        },
    )


def _load_safetensors(name: str) -> "dict | None":
    """Load a converted safetensors state_dict from models/, or None if absent."""
    from safetensors.torch import load_file

    path = MODELS_DIR / f"{name}.safetensors"
    if not path.exists():
        print(f"  skipping {name}: {path} absent — run tools/fetch_and_convert.py first")
        return None
    return load_file(str(path))


def make_pipnet_fixture() -> "dict | None":
    """PIPNet ResNet-18 (MIT) real weights on a seed-7 torch.randn(1,3,256,256).

    Loads the converted models/landmark-pipnet.safetensors into the vendored
    reference arch and records the five raw head score maps.
    """
    import _pipnet_ref

    sd = _load_safetensors("landmark-pipnet")
    if sd is None:
        return None
    net = _pipnet_ref.build_pipnet()
    net.load_state_dict(sd, strict=True)
    x = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        cls, ox, oy, nx, ny = net(x)
    return _save_fixture(
        "landmark-pipnet-real",
        {"input": x, "cls": cls, "x": ox, "y": oy, "nb_x": nx, "nb_y": ny},
        {
            "net": "torchlm/PIPNet ResNet-18 (MIT) 68-point face landmark net: stock "
                   "torchvision ResNet-18 stem+trunk, five parallel 1x1 conv heads, raw "
                   "score maps (no sigmoid)",
            "weights": "real:models/landmark-pipnet.safetensors",
            "seed": None,
            "input_seed": 7,
            "input_domain": "torch.randn(1,3,256,256)",
            "notes": "input is HxW = 256x256 (PIPNet 300W crop size); outputs are the "
                     "five raw ResNet-18 head score maps before any sigmoid/decoding",
        },
    )


# --------------------------------------------------------------------------- irn50


def make_irn50_fixture() -> dict:
    """IRN-50 with fully synthetic deterministic weights (seed 9012).

    The upstream module can only be constructed *through* its npy weights dict
    (the __conv/__batch_normalization/__dense helpers copy from
    _weights_dict[name] at construction time: conv needs 'weights' (+optional
    'bias'), bn needs 'mean'+'var' (+optional 'scale','bias'), dense needs
    'weights' (+optional 'bias')). Instead of fabricating that dict we swap the
    three helpers for plain layer constructors (identical nn.* calls, no
    copy), then overwrite every parameter/buffer via the shared seeded recipe
    so the weights are spec'd exactly like fixtures B/C. The saved safetensors
    uses the module's state_dict keys — the same keys fetch_and_convert.py
    --irn50 produces for the real npy.
    """
    import torch.nn as nn
    import face_feature.irn50_pytorch as irn_mod

    cls = irn_mod.irn50_pytorch

    def _conv(dim, name, **kw):
        assert dim == 2, name
        return nn.Conv2d(**kw)

    def _bn(dim, name, **kw):
        return nn.BatchNorm1d(**kw) if dim in (0, 1) else nn.BatchNorm2d(**kw)

    def _dense(name, **kw):
        return nn.Linear(**kw)

    names = ("_irn50_pytorch__conv", "_irn50_pytorch__batch_normalization",
             "_irn50_pytorch__dense")
    orig = {n: getattr(cls, n) for n in names}
    try:
        cls._irn50_pytorch__conv = staticmethod(_conv)
        cls._irn50_pytorch__batch_normalization = staticmethod(_bn)
        cls._irn50_pytorch__dense = staticmethod(_dense)
        net = cls(None)  # load_weights(None) -> None; helpers never touch it
    finally:
        for n, v in orig.items():
            setattr(cls, n, v)

    fill_random_state(net, 9012)
    net.eval()
    wdesc = save_state_dict_safetensors(net.state_dict(), FIXTURES_DIR / "irn50-rand.safetensors")
    x = pseudo_image(9013, (1, 3, 128, 128), "unit256")
    with torch.no_grad():
        out = net(x)
    return _save_fixture(
        "irn50-rand",
        {"input": x, "embedding": out},
        {
            "net": "upstream face_feature/irn50_pytorch.py irn50_pytorch (inception-"
                   "resnet-50 variant, bn_eps=9.999999747378752e-06, asymmetric pads, "
                   "final 512-d FC -> maxout over halves -> 256-d)",
            "weights": "irn50-rand.safetensors",
            "seed": 9012,
            "weight_recipe": RECIPE,
            "input_seed": 9013,
            "input_domain": "randint(0,256) / 256",
            "num_weight_tensors": len(wdesc),
            "notes": "output is the pre-normalization 256-d maxout embedding "
                     "(eltwise max of bn_fc1[:, :256] and bn_fc1[:, 256:])",
        },
    )


# --------------------------------------------------------------------------- embedder


def make_embedder_fixture(paths) -> dict:
    xm = load_py_module(paths["xiaoccer_model.py"], "rvface_xiaoccer_mfn")
    net = xm.MobileFacenet()
    ck = torch.load(paths["embedder.ckpt"], map_location="cpu", weights_only=True)
    net.load_state_dict(strip_module_prefix(ck["net_state_dict"]), strict=True)
    net.eval()
    # NOTE: the checkpoint is 112x96 (GDC kernel 7x6), NOT 112x112
    x = pseudo_image(2025, (1, 3, 112, 96), "pm1")
    with torch.no_grad():
        out = net(x)
    return _save_fixture(
        "embedder-mfn-real",
        {"input": x, "embedding": out},
        {
            "net": "Xiaoccer/MobileFaceNet_Pytorch core/model.py MobileFacenet(): "
                   "bottleneck-style MobileFaceNet, PReLU, GDC kernel 7x6, 128-d",
            "weights": "real:models/embedder-mfn.safetensors",
            "seed": None,
            "input_seed": 2025,
            "input_domain": "(randint(0,256) - 127.5) / 128",
            "notes": "input is HxW = 112x96 (SphereFace-aligned crop size); output is "
                     "the raw (not L2-normalized) 128-d embedding",
        },
    )


def make_embedder_foamliu_fixture(paths) -> dict:
    fm = common.load_foamliu_mfn(paths["foamliu_mobilefacenet.py"])
    net = fm.MobileFaceNet()
    sd = strip_module_prefix(
        torch.load(paths["foamliu_embedder.pt"], map_location="cpu", weights_only=True)
    )
    net.load_state_dict(sd, strict=True)
    net.eval()
    x = pseudo_image(2026, (1, 3, 112, 112), "imagenet")
    with torch.no_grad():
        out = net(x)
    return _save_fixture(
        "embedder-foamliu-real",
        {"input": x, "embedding": out},
        {
            "net": "foamliu/MobileFaceNet mobilefacenet.py MobileFaceNet(): "
                   "MobileNetV2-style inverted residuals, ReLU6 (plain ReLU in the "
                   "dw_conv stem), GDConv 7x7, biased 1x1 head conv + BN, 128-d",
            "weights": "real:models/embedder-foamliu.safetensors",
            "seed": None,
            "input_seed": 2026,
            "input_domain": "(randint(0,256)/255 - imagenet_mean) / imagenet_std",
            "notes": "input is HxW = 112x112; output is the raw (not L2-normalized) "
                     "128-d embedding",
        },
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure(
        "detector.pth", "embedder.ckpt",
        "xiaoccer_model.py",
        "foamliu_embedder.pt", "foamliu_mobilefacenet.py",
        "sdk/mb_tiny.py", "sdk/ssd.py", "sdk/mb_tiny_fd.py", "sdk/predictor.py",
        "sdk/data_preprocessing.py", "sdk/transforms.py", "sdk/fd_config.py",
        "sdk/box_utils.py", "sdk/misc.py", "sdk/MobileFaceNet.py",
        "sdk/irn50_pytorch.py",
    )
    common.install_import_stubs()
    common.add_sdk_to_path()

    torch.set_grad_enabled(False)
    print("generating fixtures:")
    entries = []
    entries += make_detector_fixtures(paths)
    entries.append(make_landmark64_fixture())
    # PIPNet real-weight fixture loads the converted safetensors; skipped (None)
    # when fetch_and_convert.py has not produced it yet.
    pipnet = make_pipnet_fixture()
    if pipnet is not None:
        entries.append(pipnet)
    entries.append(make_irn50_fixture())
    entries.append(make_embedder_fixture(paths))
    entries.append(make_embedder_foamliu_fixture(paths))

    write_json(FIXTURES_DIR / "INDEX.json", {
        "tolerance_base": TOLERANCE,
        "tolerance_rule": "per fixture: max|delta| <= 1e-4 * max(1, max|output|), absolute",
        "weight_recipe": RECIPE,
        "recipe_doc": "../naming.md",
        "fixtures": entries,
    })
    print(f"wrote {FIXTURES_DIR / 'INDEX.json'} ({len(entries)} fixtures)")


if __name__ == "__main__":
    main()
