// remove-bg.js — Eliminación de fondo con RMBG-1.4 (BRIA AI, IS-Net) y sombra
// sintética de caída (drop shadow) a partir de la máscara del propio modelo.
// Réplica del pipeline oficial (briaai/RMBG-1.4, ejemplo ONNX del model card):
//   1. Reducir la imagen a la entrada fija del modelo (1024×1024, estirando),
//      normalizar: x/255 - 0.5 (mean 0.5, std 1.0).
//   2. Inferencia → máscara de logits [1,1,1024,1024].
//   3. Máscara reescalada (bilineal, float) al tamaño original y normalizada
//      min-max a [0,1] → canal alpha. El color nunca pasa por el modelo:
//      el RGB de salida es el de la imagen original a resolución completa.
// La sombra NO es un modelo de IA: se compone en canvas (silueta del alpha +
// desplazamiento + desenfoque gaussiano + opacidad) sobre fondo blanco o
// transparente. El recorte se cachea en memoria, de modo que updateShadow()
// recompone al instante al mover los controles, sin repetir la inferencia.
// La carga del modelo usa ModelLoader (js/model-loader.js): caché en memoria y
// Cache Storage (disco del usuario), ./models/ servido por la web o, si no está,
// descarga desde HuggingFace. Sesión en caché y fallback WebGPU → WASM también
// ante fallos en la primera inferencia.

const MODEL_URL = './models/rmbg-1.4.onnx';
// Respaldo remoto (HuggingFace) para despliegues estáticos sin los .onnx
// grandes en el servidor (Azure Static Web Apps).
const MODEL_REMOTE_URL = 'https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx';
const MODEL_MIN_BYTES = 170_000_000; // real: ~176 MB (fp32)
const INPUT_SIZE = 1024;             // entrada fija: [1,3,1024,1024]

let session = null;
let backend = null;
let forceWasm = false;

// Último recorte calculado por addShadow(): { canvas, w, h }. Lo reutiliza
// updateShadow() para recomponer la sombra sin volver a ejecutar el modelo.
let ultimoRecorte = null;
// URL del último render de sombra, para revocarla al reemplazarla (los
// sliders pueden generar decenas de blobs en una sola pasada).
let ultimaSombraUrl = null;

function webGpuSupported() {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

async function getSession(onProgreso = null) {
    if (session) return session;
    const bytes = await ModelLoader.cargarModelo({
        url: MODEL_URL,
        remoteUrl: MODEL_REMOTE_URL,
        minBytes: MODEL_MIN_BYTES,
        onProgreso
    });
    const providers = (!forceWasm && webGpuSupported()) ? ['webgpu', 'wasm'] : ['wasm'];
    let lastError = null;
    for (const ep of providers) {
        try {
            session = await ort.InferenceSession.create(bytes, {
                executionProviders: [ep],
                graphOptimizationLevel: 'all'
            });
            backend = ep;
            return session;
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

// Ejecuta fn() con reintento en WASM (CPU) si la sesión WebGPU se crea bien
// pero falla en la primera inferencia: se libera y se reintenta todo.
async function conReintentoWasm(fn) {
    try {
        return await fn();
    } catch (err) {
        if (forceWasm || backend !== 'webgpu') throw err;
        console.warn('WebGPU falló durante la inferencia; reintentando con WASM (CPU).', err);
        forceWasm = true;
        try { await session?.release?.(); } catch { /* mejor esfuerzo */ }
        session = null;
        backend = null;
        try {
            return await fn();
        } catch (errWasm) {
            throw new Error(`Falló WebGPU (${err.message}) y también WASM (${errWasm.message})`);
        }
    }
}

// Pipeline común: inferencia de RMBG-1.4 + postprocesado oficial → lienzo con
// el sujeto recortado (RGB original + alpha de la máscara) a tamaño original.
// Devuelve { canvas, w, h }.
async function calcularRecorte(bytes, mimeType, progressHelper) {
    const onProgreso = progressHelper
        ? p => progressHelper.invokeMethodAsync('OnEstado', p < 0
            ? 'Descargando modelo de eliminación de fondo (solo la primera vez)…'
            : `Descargando modelo de eliminación de fondo (solo la primera vez)… ${p}%`)
        : null;
    const sess = await getSession(onProgreso);

    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const w = bitmap.width, h = bitmap.height;

    // Entrada del modelo: RGB estirado a 1024×1024, normalizado x/255 - 0.5
    // (mean [0.5,0.5,0.5], std [1,1,1], como el pipeline oficial).
    const S = INPUT_SIZE;
    const small = new OffscreenCanvas(S, S);
    const smallCtx = small.getContext('2d', { willReadFrequently: true });
    smallCtx.imageSmoothingEnabled = true;
    smallCtx.imageSmoothingQuality = 'high';
    smallCtx.drawImage(bitmap, 0, 0, S, S);
    const smallData = smallCtx.getImageData(0, 0, S, S).data;

    const ns = S * S;
    const chw = new Float32Array(3 * ns);
    for (let i = 0; i < ns; i++) {
        chw[i] = smallData[i * 4] / 255 - 0.5;
        chw[ns + i] = smallData[i * 4 + 1] / 255 - 0.5;
        chw[2 * ns + i] = smallData[i * 4 + 2] / 255 - 0.5;
    }

    if (progressHelper)
        await progressHelper.invokeMethodAsync('OnEstado', 'Detectando el sujeto…');
    const res = await sess.run({
        [sess.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, S, S])
    });
    const logits = res[sess.outputNames[0]].data; // [1, 1, S, S]

    // Máscara a resolución original (bilineal float) + min-max, como el
    // postprocesado oficial: (x - min) / (max - min).
    const mask = resizeBilinear(logits, S, S, w, h);
    let mi = Infinity, ma = -Infinity;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i] < mi) mi = mask[i];
        if (mask[i] > ma) ma = mask[i];
    }
    const rango = ma - mi;

    // RGB original a resolución completa + alpha de la máscara.
    const src = new OffscreenCanvas(w, h);
    const srcCtx = src.getContext('2d', { willReadFrequently: true });
    srcCtx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const outImg = srcCtx.getImageData(0, 0, w, h);
    const n = w * h;
    for (let i = 0; i < n; i++) {
        const a = rango > 0 ? (mask[i] - mi) / rango : 0;
        outImg.data[i * 4 + 3] = Math.round(a * 255);
    }

    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext('2d').putImageData(outImg, 0, 0);
    return { canvas, w, h };
}

// ---------------- Sombra sintética ----------------

// Compone fondo + sombra + sujeto en un lienzo w×h.
// p: { angulo (°), distancia (px), difuminado (px), opacidad (0-100),
//      fondo ("blanco" | "transparente") }.
function renderSombra(recorte, p) {
    const { canvas: recorteCanvas, w, h } = recorte;
    const ang = (p.angulo ?? 45) * Math.PI / 180;
    const dist = Math.max(0, p.distancia ?? 25);
    const blur = Math.max(0, p.difuminado ?? 30);
    const opac = Math.min(100, Math.max(0, p.opacidad ?? 60)) / 100;
    const dx = Math.cos(ang) * dist, dy = Math.sin(ang) * dist;

    // Silueta negra con el alpha del recorte.
    const sil = new OffscreenCanvas(w, h);
    const silCtx = sil.getContext('2d');
    silCtx.drawImage(recorteCanvas, 0, 0);
    silCtx.globalCompositeOperation = 'source-in';
    silCtx.fillStyle = '#000';
    silCtx.fillRect(0, 0, w, h);

    const out = new OffscreenCanvas(w, h);
    const outCtx = out.getContext('2d');
    if (p.fondo !== 'transparente') {
        outCtx.fillStyle = '#fff';
        outCtx.fillRect(0, 0, w, h);
    }

    // La sombra se dibuja en un lienzo con margen para que el desenfoque no
    // quede cortado en los bordes de la imagen.
    const pad = Math.ceil(blur * 3 + dist + 2);
    const sh = new OffscreenCanvas(w + 2 * pad, h + 2 * pad);
    const shCtx = sh.getContext('2d');
    if (blur > 0) shCtx.filter = `blur(${blur}px)`;
    shCtx.globalAlpha = opac;
    shCtx.drawImage(sil, pad + dx, pad + dy);
    outCtx.drawImage(sh, -pad, -pad);

    outCtx.drawImage(recorteCanvas, 0, 0);
    return out;
}

async function sombraPngUrl(recorte, params) {
    const blob = await renderSombra(recorte, params ?? {}).convertToBlob({ type: 'image/png' });
    if (ultimaSombraUrl) URL.revokeObjectURL(ultimaSombraUrl);
    ultimaSombraUrl = URL.createObjectURL(blob);
    return ultimaSombraUrl;
}

// ---------------- Puntos de entrada ----------------

// Quita el fondo manteniendo el tamaño original; devuelve PNG con transparencia.
// progressHelper (DotNetObjectReference): OnEstado(string) para la fase.
// Devuelve { resultUrl, backend, width, height }.
export async function removeBackground(bytes, mimeType, progressHelper) {
    return conReintentoWasm(async () => {
        const recorte = await calcularRecorte(bytes, mimeType, progressHelper);
        const blob = await recorte.canvas.convertToBlob({ type: 'image/png' });
        return {
            resultUrl: URL.createObjectURL(blob),
            backend,
            width: recorte.w,
            height: recorte.h
        };
    });
}

// Quita el fondo y añade una sombra sintética configurable; el recorte queda
// cacheado para que updateShadow() recomponga sin repetir la inferencia.
// Devuelve { resultUrl, backend, width, height }.
export async function addShadow(bytes, mimeType, progressHelper, params) {
    return conReintentoWasm(async () => {
        ultimoRecorte = await calcularRecorte(bytes, mimeType, progressHelper);
        return {
            resultUrl: await sombraPngUrl(ultimoRecorte, params),
            backend,
            width: ultimoRecorte.w,
            height: ultimoRecorte.h
        };
    });
}

// Recompone la sombra del último recorte con nuevos parámetros (sliders).
// No ejecuta el modelo. Devuelve { resultUrl, width, height }.
export async function updateShadow(params) {
    if (!ultimoRecorte)
        throw new Error('No hay ningún recorte calculado: procesa primero la imagen.');
    return {
        resultUrl: await sombraPngUrl(ultimoRecorte, params),
        width: ultimoRecorte.w,
        height: ultimoRecorte.h
    };
}
