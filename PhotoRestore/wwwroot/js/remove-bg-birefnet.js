// remove-bg-birefnet.js — Eliminación de fondo con BiRefNet_lite en dos
// variantes (el modelo base es MIT):
//
//   - removeBackgroundBiRefNet:    512×512 (Phoenix-ActuallyFree, 185 MB)
//   - removeBackgroundBiRefNetMax: 1024×1024 (onnx-community oficial, 214 MB)
//
// ¿Por qué dos variantes y no el BiRefNet-general completo?
//   - BiRefNet-general fp32 (~928 MB) no cabe en el heap WASM de
//     onnxruntime-web ("failed to allocate a buffer of size 972666916").
//   - BiRefNet-general fp16 (~467 MB) exige la feature WebGPU "shader-f16";
//     en GPUs que no la exponen la sesión no se crea ("Program Transpose
//     requires f16 but the device does not support it").
//   - BiRefNet_lite oficial a 1024×1024: carga bien, pero la INFERENCIA
//     necesita ~8,7 GB de arena (verificado con onnxruntime-node en CPU: el
//     grafo de 16.400 nodos de este export explota en memoria con las convs
//     deformables emuladas con GatherND/ScatterND) → std::bad_alloc en WASM
//     (heap máx. 4 GB). Solo es usable con WebGPU en GPUs con bastante VRAM.
//   - El re-exporte a 512×512 tiene un grafo mucho más limpio (2.213 nodos) y
//     necesita ~2,2 GB: funciona con WebGPU y con el fallback WASM/CPU.
//     Verificado contra el oficial a 1024 con foto real: correlación de
//     máscaras 0,9989 (misma salida, algo menos de detalle fino).
//
// Diferencias de pipeline entre variantes:
//   - Ambas: RGB estirado a su entrada fija y normalización ImageNet
//     (x/255 - mean) / std, mean [0.485,0.456,0.406], std [0.229,0.224,0.225].
//   - 512: la salida YA lleva sigmoid (el grafo termina en Sigmoid) → es
//     probabilidad [0,1] directamente.
//   - 1024 (oficial): la salida son logits → hay que aplicar sigmoid en JS.
//   - Reescalado bilineal de la máscara al tamaño original → canal alpha.
//     El color nunca pasa por el modelo: el RGB de salida es el original.
// La carga usa ModelLoader (js/model-loader.js): caché en memoria y Cache
// Storage, ./models/ servido por la web o respaldo remoto en HuggingFace.
// Sesiones en caché y fallback WebGPU → WASM ante fallos en la primera
// inferencia.

const MODELOS = {
    lite512: {
        url: './models/birefnet-lite-512.onnx',
        remoteUrl: 'https://huggingface.co/Phoenix-ActuallyFree/BiRefNet_lite_512-ONNX/resolve/main/onnx/model.onnx',
        minBytes: 190_000_000,   // real: 193.514.682 B (fp32)
        inputSize: 512,
        salidaConSigmoid: true,
        etiquetaMB: '~185 MB'
    },
    lite1024: {
        url: './models/birefnet-lite.onnx',
        remoteUrl: 'https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx',
        minBytes: 220_000_000,   // real: 224.005.088 B (fp32)
        inputSize: 1024,
        salidaConSigmoid: false,
        etiquetaMB: '~214 MB'
    }
};

// Sesiones y estado por variante (url -> { session, backend, forceWasm }).
const estados = new Map();

function webGpuSupported() {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

async function getSession(cfg, estado, onProgreso = null) {
    if (estado.session) return estado.session;
    const bytes = await ModelLoader.cargarModelo({
        url: cfg.url,
        remoteUrl: cfg.remoteUrl,
        minBytes: cfg.minBytes,
        onProgreso
    });
    const providers = (!estado.forceWasm && webGpuSupported()) ? ['webgpu', 'wasm'] : ['wasm'];
    let lastError = null;
    for (const ep of providers) {
        try {
            estado.session = await ort.InferenceSession.create(bytes, {
                executionProviders: [ep],
                graphOptimizationLevel: 'all'
            });
            estado.backend = ep;
            return estado.session;
        } catch (err) {
            lastError = err;
        }
    }
    throw lastError;
}

// Reescalado bilineal de un canal float (más preciso que pasar por canvas 8-bit).
function resizeBilinear(src, sw, sh, dw, dh) {
    const dst = new Float32Array(dw * dh);
    const rx = sw / dw, ry = sh / dh;
    for (let y = 0; y < dh; y++) {
        const sy = (y + 0.5) * ry - 0.5;
        const y0 = Math.max(0, Math.floor(sy)), y1 = Math.min(sh - 1, y0 + 1);
        const fy = Math.min(Math.max(sy - y0, 0), 1);
        for (let x = 0; x < dw; x++) {
            const sx = (x + 0.5) * rx - 0.5;
            const x0 = Math.max(0, Math.floor(sx)), x1 = Math.min(sw - 1, x0 + 1);
            const fx = Math.min(Math.max(sx - x0, 0), 1);
            const v00 = src[y0 * sw + x0], v10 = src[y0 * sw + x1];
            const v01 = src[y1 * sw + x0], v11 = src[y1 * sw + x1];
            dst[y * dw + x] = (v00 * (1 - fx) + v10 * fx) * (1 - fy) +
                              (v01 * (1 - fx) + v11 * fx) * fy;
        }
    }
    return dst;
}

// ---------------- Puntos de entrada ----------------

// Quita el fondo manteniendo el tamaño original; devuelve PNG con transparencia.
// progressHelper (DotNetObjectReference): OnEstado(string) para la fase.
// Devuelve { resultUrl, backend, width, height }.
export async function removeBackgroundBiRefNet(bytes, mimeType, progressHelper) {
    return await removeBackgroundConFallback(MODELOS.lite512, bytes, mimeType, progressHelper);
}

// Variante a 1024×1024 (más detalle fino). OJO: la inferencia necesita ~8,7 GB;
// en la práctica solo funciona con WebGPU en GPUs con bastante VRAM.
export async function removeBackgroundBiRefNetMax(bytes, mimeType, progressHelper) {
    return await removeBackgroundConFallback(MODELOS.lite1024, bytes, mimeType, progressHelper);
}

async function removeBackgroundConFallback(cfg, bytes, mimeType, progressHelper) {
    let estado = estados.get(cfg.url);
    if (!estado) {
        estado = { session: null, backend: null, forceWasm: false };
        estados.set(cfg.url, estado);
    }
    try {
        return await removeBackgroundInterno(cfg, estado, bytes, mimeType, progressHelper);
    } catch (err) {
        // La sesión WebGPU puede crearse bien pero fallar en la primera
        // inferencia: se libera y se reintenta todo con WASM (CPU).
        if (estado.forceWasm || estado.backend !== 'webgpu') throw err;
        console.warn('WebGPU falló durante la inferencia; reintentando con WASM (CPU).', err);
        estado.forceWasm = true;
        try { await estado.session?.release?.(); } catch { /* mejor esfuerzo */ }
        estado.session = null;
        estado.backend = null;
        try {
            return await removeBackgroundInterno(cfg, estado, bytes, mimeType, progressHelper);
        } catch (errWasm) {
            // err puede no ser un Error (p. ej. un número o un objeto del EP):
            // String() evita mensajes "undefined" que ocultan la causa real.
            throw new Error(`Falló WebGPU (${err?.message ?? String(err)}) y también WASM (${errWasm?.message ?? String(errWasm)})`);
        }
    }
}

async function removeBackgroundInterno(cfg, estado, bytes, mimeType, progressHelper) {
    const onProgreso = progressHelper
        ? p => progressHelper.invokeMethodAsync('OnEstado', p < 0
            ? `Descargando modelo BiRefNet (solo la primera vez, ${cfg.etiquetaMB})…`
            : `Descargando modelo BiRefNet (solo la primera vez, ${cfg.etiquetaMB})… ${p}%`)
        : null;
    const sess = await getSession(cfg, estado, onProgreso);

    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const w = bitmap.width, h = bitmap.height;

    // Entrada del modelo: RGB estirado a la entrada fija, normalización
    // ImageNet (x/255 - mean) / std.
    const S = cfg.inputSize;
    const small = new OffscreenCanvas(S, S);
    const smallCtx = small.getContext('2d', { willReadFrequently: true });
    smallCtx.imageSmoothingEnabled = true;
    smallCtx.imageSmoothingQuality = 'high';
    smallCtx.drawImage(bitmap, 0, 0, S, S);
    const smallData = smallCtx.getImageData(0, 0, S, S).data;

    const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
    const ns = S * S;
    const chw = new Float32Array(3 * ns);
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < ns; i++)
            chw[c * ns + i] = (smallData[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }

    if (progressHelper)
        await progressHelper.invokeMethodAsync('OnEstado', 'Detectando el sujeto…');
    const res = await sess.run({
        [sess.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, S, S])
    });
    const salida = res[sess.outputNames[0]].data; // [1, 1, S, S]

    // Máscara de probabilidad: el export 512 ya incluye el sigmoid en el
    // grafo; el oficial a 1024 devuelve logits y hay que aplicarlo aquí.
    // Sin min-max en ninguno de los dos: no forma parte del pipeline oficial.
    let mask;
    if (cfg.salidaConSigmoid) {
        mask = resizeBilinear(salida, S, S, w, h);
    } else {
        const sig = new Float32Array(ns);
        for (let i = 0; i < ns; i++)
            sig[i] = 1 / (1 + Math.exp(-salida[i]));
        mask = resizeBilinear(sig, S, S, w, h);
    }

    // RGB original a resolución completa + alpha de la máscara.
    const src = new OffscreenCanvas(w, h);
    const srcCtx = src.getContext('2d', { willReadFrequently: true });
    srcCtx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const outImg = srcCtx.getImageData(0, 0, w, h);
    const n = w * h;
    for (let i = 0; i < n; i++) {
        const a = Math.min(Math.max(mask[i], 0), 1);
        outImg.data[i * 4 + 3] = Math.round(a * 255);
    }

    const out = new OffscreenCanvas(w, h);
    out.getContext('2d').putImageData(outImg, 0, 0);
    const blob = await out.convertToBlob({ type: 'image/png' });
    return {
        resultUrl: URL.createObjectURL(blob),
        backend: estado.backend,
        width: w,
        height: h
    };
}
