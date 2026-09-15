// upscaler.js — Restauración con Real-ESRGAN en el navegador (ONNX Runtime Web).
// El modelo siempre trabaja a 4x internamente; la salida se reescala al factor
// elegido (1, 2 o 4). Toda la inferencia se ejecuta en el dispositivo:
// la imagen nunca se sube a ningún servidor.
//
// El modelo NO se pasa a ONNX Runtime por URL: se descarga manualmente con
// fetch (cache: 'no-cache', con un reintento), se valida su tamaño y se crea la
// sesión desde los bytes. Así los errores de descarga llegan a la UI con
// mensajes claros en lugar de un "Failed to fetch" opaco de ort.

const MODEL_URL = './models/realesrgan-x4.onnx';
const MODEL_MIN_BYTES = 1_000_000; // el .onnx real ocupa 4.866.429 bytes
const SCALE = 4;    // factor nativo del modelo Real-ESRGAN x4
const TILE = 128;   // tamaño del tile de entrada (px)
const OVERLAP = 10; // solape entre tiles (px de entrada) para evitar costuras

let session = null;
let backend = null;

export function webGpuSupported() {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

// Descarga el modelo con un reintento y mensajes de error autoexplicativos.
async function fetchModel() {
    let lastError = null;
    for (let intento = 1; intento <= 2; intento++) {
        try {
            console.log(`Descargando modelo (intento ${intento})…`);
            const resp = await fetch(MODEL_URL, { cache: 'no-cache' });
            if (!resp.ok)
                throw new Error(`Error del servidor al descargar el modelo: HTTP ${resp.status}`);
            const buffer = await resp.arrayBuffer();
            if (buffer.byteLength < MODEL_MIN_BYTES)
                throw new Error(`El modelo descargado parece incompleto (${buffer.byteLength} bytes); recarga la página e inténtalo de nuevo.`);
            return buffer;
        } catch (err) {
            lastError = err;
            console.warn(`Fallo al descargar el modelo (intento ${intento}).`, err);
            if (intento < 2)
                await new Promise(r => setTimeout(r, 800));
        }
    }
    if (lastError instanceof TypeError) // error de red de fetch
        throw new Error(`No se pudo descargar el modelo (¿el servidor sigue corriendo?): ${lastError.message}`);
    throw lastError;
}

// Crea la sesión una sola vez a partir de los bytes del modelo.
// Se intenta WebGPU primero y se cae a WASM (CPU) de forma explícita,
// para poder informar del motor que realmente se usa.
async function getSession() {
    if (session) return session;
    const modelBytes = new Uint8Array(await fetchModel());
    const providers = webGpuSupported() ? ['webgpu', 'wasm'] : ['wasm'];
    let lastError = null;
    for (const ep of providers) {
        try {
            session = await ort.InferenceSession.create(modelBytes, {
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

// URL de objeto para previsualizar un byte[] recibido de Blazor.
export function createObjectUrl(bytes, mimeType) {
    return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

export function revokeObjectUrl(url) {
    if (url) URL.revokeObjectURL(url);
}

// Dimensiones naturales de la imagen, para mostrar el tamaño de salida previsto.
export async function getImageSize(bytes, mimeType) {
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return size;
}

// Restaura la imagen procesando por tiles para acotar la memoria.
// scale es el factor de salida deseado (1, 2 o 4); el modelo corre siempre a 4x
// y el resultado compuesto se reescala a w*scale × h*scale si hace falta.
// progressHelper es un DotNetObjectReference con un método [JSInvokable]
// OnTileDone(int done, int total) que se invoca tras cada tile.
// Devuelve { resultUrl, backend, width, height } (dimensiones de salida).
export async function upscale(bytes, mimeType, progressHelper, scale) {
    if (![1, 2, 4].includes(scale))
        throw new Error(`Factor de salida no válido: ${scale} (solo 1, 2 o 4)`);

    const sess = await getSession();
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const w = bitmap.width, h = bitmap.height;

    const src = new OffscreenCanvas(w, h);
    const srcCtx = src.getContext('2d', { willReadFrequently: true });
    srcCtx.drawImage(bitmap, 0, 0);
    bitmap.close();

    const dst = new OffscreenCanvas(w * SCALE, h * SCALE);
    const dstCtx = dst.getContext('2d');

    const inName = sess.inputNames[0];
    const outName = sess.outputNames[0];

    const tiles = [];
    for (let y = 0; y < h; y += TILE)
        for (let x = 0; x < w; x += TILE)
            tiles.push({ x, y });

    for (let i = 0; i < tiles.length; i++) {
        const { x, y } = tiles[i];
        // Región de entrada incluyendo el solape, recortada a los bordes de la imagen.
        const sx0 = Math.max(0, x - OVERLAP);
        const sy0 = Math.max(0, y - OVERLAP);
        const sx1 = Math.min(w, x + TILE + OVERLAP);
        const sy1 = Math.min(h, y + TILE + OVERLAP);
        const tw = sx1 - sx0, th = sy1 - sy0;

        const tile = srcCtx.getImageData(sx0, sy0, tw, th);
        const results = await sess.run({ [inName]: rgbaToTensor(tile) });
        const outImg = tensorToImageData(results[outName], tw * SCALE, th * SCALE);

        // Se descartan los márgenes de solape (ya escalados) y se pega la zona útil.
        const tmp = new OffscreenCanvas(outImg.width, outImg.height);
        tmp.getContext('2d').putImageData(outImg, 0, 0);
        const cx = (x - sx0) * SCALE, cy = (y - sy0) * SCALE;
        const cw = Math.min(TILE, w - x) * SCALE, ch = Math.min(TILE, h - y) * SCALE;
        dstCtx.drawImage(tmp, cx, cy, cw, ch, x * SCALE, y * SCALE, cw, ch);

        if (progressHelper)
            await progressHelper.invokeMethodAsync('OnTileDone', i + 1, tiles.length);
    }

    // Reescalado al factor elegido. Si la salida del modelo ya tiene el tamaño
    // objetivo (x4), se usa directamente sin remuestrear.
    const targetW = w * scale, targetH = h * scale;
    let out = dst;
    if (targetW !== dst.width || targetH !== dst.height) {
        out = new OffscreenCanvas(targetW, targetH);
        const outCtx = out.getContext('2d');
        outCtx.imageSmoothingEnabled = true;
        outCtx.imageSmoothingQuality = 'high';
        outCtx.drawImage(dst, 0, 0, targetW, targetH);
    }

    const blob = await out.convertToBlob({ type: 'image/png' });
    return {
        resultUrl: URL.createObjectURL(blob),
        backend,
        width: out.width,
        height: out.height
    };
}

// RGBA interleaved (0..255) -> tensor float32 NCHW normalizado a 0..1
function rgbaToTensor(imageData) {
    const { data, width, height } = imageData;
    const n = width * height;
    const chw = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        chw[i] = data[i * 4] / 255;
        chw[n + i] = data[i * 4 + 1] / 255;
        chw[2 * n + i] = data[i * 4 + 2] / 255;
    }
    return new ort.Tensor('float32', chw, [1, 3, height, width]);
}

// Tensor NCHW (0..1) -> ImageData RGBA (0..255), con saturación
function tensorToImageData(tensor, width, height) {
    const src = tensor.data;
    const n = width * height;
    const img = new ImageData(width, height);
    const px = img.data;
    for (let i = 0; i < n; i++) {
        px[i * 4] = clamp255(src[i]);
        px[i * 4 + 1] = clamp255(src[n + i]);
        px[i * 4 + 2] = clamp255(src[2 * n + i]);
        px[i * 4 + 3] = 255;
    }
    return img;
}

function clamp255(v) {
    v = Math.round(v * 255);
    return v < 0 ? 0 : v > 255 ? 255 : v;
}
