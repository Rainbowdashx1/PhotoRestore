"""Exporta DDColor-tiny a ONNX fp32 para `wwwroot/models/ddcolor.onnx`.

Pasos (autocontenido):
1. Clona el repo oficial de DDColor (arquitectura) en `obj/ddcolor-export/`.
2. Descarga los pesos `pytorch_model.bin` de HF `piddnad/ddcolor_paper_tiny`.
3. Construye el modelo con el `config.json` de HF y exporta a ONNX opset 17.
4. Fija `ceil_mode=0` en los AveragePool si el export los deja en 1
   (el EP WebGPU de onnxruntime-web no soporta ceil en shape computation).
5. Verifica con onnxruntime CPU contra el modelo PyTorch (misma entrada fija).

Uso: `.venv/Scripts/python.exe tools/export_ddcolor.py`
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # PhotoRestore/
REPO_DIR = ROOT / "obj" / "ddcolor-export" / "DDColor"
REPO_URL = "https://github.com/piddnad/DDColor.git"
HF_REPO = "piddnad/ddcolor_paper_tiny"
OUT_PATH = ROOT / "wwwroot" / "models" / "ddcolor.onnx"

INPUT_SHAPE = (1, 3, 512, 512)
OPSET = 17
SEED = 0
MAX_DIFF_TOL = 0.05


def clone_repo() -> None:
    if REPO_DIR.is_dir():
        print(f"Repo DDColor ya presente en {REPO_DIR}")
        return
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True
    )


def build_model() -> torch.nn.Module:
    # Solo se importa el subconjunto ligero: ddcolor.model -> basicsr.archs.ddcolor_arch_utils
    # (torch/timm/numpy; sin instalar basicsr completo ni skimage).
    sys.path.insert(0, str(REPO_DIR))
    from huggingface_hub import hf_hub_download

    from ddcolor.model import DDColor
    from ddcolor.pipeline import load_checkpoint_state_dict

    config = json.loads(Path(hf_hub_download(HF_REPO, "config.json")).read_text())
    print(f"Config HF: {config}")
    model = DDColor(**config)

    weights = hf_hub_download(HF_REPO, "pytorch_model.bin")
    state_dict = load_checkpoint_state_dict(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Claves ausentes: {missing}\nClaves inesperadas: {unexpected}")
    assert not unexpected, "el checkpoint no corresponde a la arquitectura"
    # 'mean'/'std' son buffers de normalización ImageNet con valores fijos en el código.
    assert all(k in ("mean", "std") for k in missing), f"faltan pesos: {missing}"

    model.eval()
    return model


def export_onnx(model: torch.nn.Module, x: torch.Tensor) -> None:
    kwargs = dict(
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    try:
        # Tracer TorchScript: el modelo usa hooks con estado que el exporter dynamo no captura.
        torch.onnx.export(model, x, str(OUT_PATH), dynamo=False, **kwargs)
    except TypeError:  # torch antiguo sin argumento `dynamo`
        torch.onnx.export(model, x, str(OUT_PATH), **kwargs)
    print(f"Exportado: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


def patch_ceil_mode() -> bool:
    """Fija ceil_mode=0 en AveragePool. Devuelve True si modificó el grafo."""
    import onnx

    model = onnx.load(str(OUT_PATH))
    patched = 0
    for node in model.graph.node:
        if node.op_type != "AveragePool":
            continue
        for attr in node.attribute:
            if attr.name == "ceil_mode" and attr.i == 1:
                attr.i = 0
                patched += 1
    total = sum(1 for n in model.graph.node if n.op_type == "AveragePool")
    print(f"AveragePool: {total} nodos, {patched} parcheados a ceil_mode=0")
    if patched:
        onnx.save(model, str(OUT_PATH))
    onnx.checker.check_model(onnx.load(str(OUT_PATH)))
    return patched > 0


def verify(model: torch.nn.Module, x: torch.Tensor) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"ONNX input:  {inp.name} {inp.shape} {inp.type}")
    print(f"ONNX output: {out.name} {out.shape} {out.type}")
    assert inp.name == "input" and inp.shape == [1, 3, 512, 512]
    assert out.name == "output" and out.shape == [1, 2, 512, 512]

    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(["output"], {"input": x.numpy()})[0]
    diff = np.abs(ref - got)
    print(f"max abs diff = {diff.max():.6f}, mean abs diff = {diff.mean():.3e}")
    assert diff.max() < MAX_DIFF_TOL, "el ONNX diverge del PyTorch oficial"


def main() -> None:
    clone_repo()
    model = build_model()

    # Entrada fija: RGB 0..1 (el pipeline oficial pasa el gris reconstruido vía Lab).
    torch.manual_seed(SEED)
    x = torch.rand(INPUT_SHAPE)

    export_onnx(model, x)
    patch_ceil_mode()
    verify(model, x)
    print("OK")


if __name__ == "__main__":
    main()
