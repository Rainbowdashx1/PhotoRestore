# Modelos de PhotoRestore

Modelos ONNX servidos estáticamente por la app. Toda la inferencia es local
(navegador, ONNX Runtime Web con WebGPU → WASM).

| Archivo | Modelo | Tamaño | Origen |
|---|---|---|---|
| `realesrgan-x4.onnx` | Real-ESRGAN-General-x4v3 (SRVGG, x4 con denoise) | 4.866.429 B | https://huggingface.co/Samo629/real-esrgan-onnx (`realesr-general-x4v3-dynfix.onnx`) |
| `scrfd-2.5g.onnx` | SCRFD 2.5G bnkps (detección de caras + 5 puntos faciales) | 3.291.737 B | https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX (`2.5g_bnkps.onnx`, Apache 2.0, de insightface) |
| `gfpgan-v1.4.onnx` | GFPGAN v1.4 (restauración de caras, fp32) | 340.357.025 B | https://huggingface.co/HowToSD/GFPGAN-ONNX (exporte fiel del .pth oficial TencentARC; Apache 2.0 + licencias de terceros, posible restricción de uso comercial) |
| `ddcolor.onnx` | DDColor-tiny (colorización B/N → color, fp32) | 270.255.132 B | **Exporte propio** (opset 17) de `piddnad/ddcolor_paper_tiny` (https://huggingface.co/piddnad/ddcolor_paper_tiny, arquitectura de https://github.com/piddnad/DDColor; Apache-2.0) |
| `rmbg-1.4.onnx` | RMBG-1.4 (eliminación de fondo, IS-Net, fp32) | 176.153.355 B | https://huggingface.co/briaai/RMBG-1.4 (`onnx/model.onnx`; licencia bria-rmbg-1.4, **solo uso no comercial**) |

## Detalles de tensores

- `realesrgan-x4.onnx` (opset 17): input `input` [batch, 3, height, width] fp32
  dinámico, RGB 0..1 → output `output` [batch, 3, outH, outW] (x4).
- `scrfd-2.5g.onnx` (opset 11): input `input.1` [1, 3, 640, 640] fp32, BGR
  normalizado a [-1, 1] con letterbox 640×640 → 9 salidas: `score_8/16/32`
  (**probabilidades**: el grafo incluye los `Sigmoid`), `bbox_8/16/32`
  (distancias × stride), `kps_8/16/32` (5 puntos × stride); 2 anclas por celda.
- `gfpgan-v1.4.onnx` (opset 16): input `input` [batch, 3, 512, 512] fp32, RGB
  normalizado a [-1, 1], cara alineada a 512×512 (plantilla ArcFace ×4) →
  output `output` [batch, 3, 512, 512] en [-1, 1].
- `ddcolor.onnx` (opset 17): input `input` [1, 3, 512, 512] fp32, RGB 0..1 del
  gris reconstruido vía Lab (sin normalización, `do_normalize=False` como el
  pipeline oficial) → output `output` [1, 2, 512, 512] con la croma AB en
  convención float de OpenCV (a,b ~[-127,127]). Pipeline completo: L de la
  imagen original a resolución completa + AB reescalado bilineal → Lab → RGB.
  Verificado contra el PyTorch oficial (foto real, entrada gris vía Lab):
  max abs diff = 0.018, mean = 3e-5, corr ≈ 1.0. Sus 4 AveragePool llevan
  `ceil_mode=0` (sin problema WebGPU). El export de Qualcomm (`image` 256×256)
  se descartó: lleva la normalización ImageNet integrada y diverge del oficial
  (corr 0.64); el de edgetools es fiel pero fp16 (rompería el fallback WASM).
- `rmbg-1.4.onnx` (opset 17): input [1, 3, 1024, 1024] fp32, RGB estirado a
  1024×1024 y normalizado x/255 − 0.5 (mean [0.5,0.5,0.5], std [1,1,1], como
  el ejemplo oficial del model card) → output [1, 1, 1024, 1024] con logits de
  la máscara. Postprocesado oficial: reescalado bilineal al tamaño original +
  min-max a [0,1] → canal alpha; el RGB de salida es el de la imagen original
  (nunca pasa por el modelo). Probado en navegador por el usuario.

## Notas

- `gfpgan-v1.4.onnx` y `rmbg-1.4.onnx` están en `.gitignore` por su tamaño;
  `ddcolor.onnx` sí está commiteado aunque la nota histórica diga lo contrario.
  Las URLs de descarga están en la tabla de arriba. Todos los modelos se
  mantienen en fp32 (no fp16) porque el EP
  WASM/CPU de ONNX Runtime Web no tiene kernels fp16 para Conv, y el fallback
  WebGPU → WASM debe seguir funcionando. Sus ops son todos comunes y soportados
  por el EP WebGPU (Conv, Resize, LeakyRelu, Gemm…).
- `scrfd-2.5g.onnx` está **parcheado**: los nodos `AveragePool_25/52/69` tenían
  `ceil_mode=1`, que el EP WebGPU de onnxruntime-web no soporta ("using ceil()
  in shape computation is not yet supported"). Sus entradas reales son
  160×160, 80×80 y 40×40 (kernel 2, stride 2, pads 0, dims pares), así que
  floor ≡ ceil y se fijó `ceil_mode=0` sin cambio semántico (verificado con
  onnxruntime CPU: diff máx. = 0.0 en las 9 salidas con entrada aleatoria fija,
  seed 0). El original está en `PhotoRestore/models-backup/scrfd-2.5g-original.onnx`
  (excluido de git).
- El alineado usa la plantilla ArcFace 112×112
  `[[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]]`
  escalada ×4, con transformación de semejanza (Umeyama) calculada en JS
  (`wwwroot/js/face-restore.js`).
