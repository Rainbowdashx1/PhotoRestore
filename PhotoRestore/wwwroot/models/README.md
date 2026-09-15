# Modelos de PhotoRestore

Modelos ONNX servidos estáticamente por la app. Toda la inferencia es local
(navegador, ONNX Runtime Web con WebGPU → WASM).

| Archivo | Modelo | Tamaño | Origen |
|---|---|---|---|
| `realesrgan-x4.onnx` | Real-ESRGAN-General-x4v3 (SRVGG, x4 con denoise) | 4.866.429 B | https://huggingface.co/Samo629/real-esrgan-onnx (`realesr-general-x4v3-dynfix.onnx`) |
| `scrfd-2.5g.onnx` | SCRFD 2.5G bnkps (detección de caras + 5 puntos faciales) | 3.291.737 B | https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX (`2.5g_bnkps.onnx`, Apache 2.0, de insightface) |
| `gfpgan-v1.4.onnx` | GFPGAN v1.4 (restauración de caras, fp32) | 340.357.025 B | https://huggingface.co/HowToSD/GFPGAN-ONNX (exporte fiel del .pth oficial TencentARC; Apache 2.0 + licencias de terceros, posible restricción de uso comercial) |

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

## Notas

- `gfpgan-v1.4.onnx` está en `.gitignore` por su tamaño; para regenerarlo,
  descargar del enlace de arriba. Se mantiene en fp32 (no fp16) porque el EP
  WASM/CPU de ONNX Runtime Web no tiene kernels fp16 para Conv, y el fallback
  WebGPU → WASM debe seguir funcionando. Sus ops son todos comunes y soportados
  por el EP WebGPU (Conv, Resize, LeakyRelu, Gemm…); no contiene AveragePool.
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
