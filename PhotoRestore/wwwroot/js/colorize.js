// colorize.js — Colorización de fotos en blanco y negro con DDColor (ICCV 2023).
// Réplica del pipeline oficial (piddnad/DDColor, convención float de OpenCV:
// L ∈ [0,100], a,b ∈ ~[-127,127], D65, gamma sRGB):
//   1. L de la imagen ORIGINAL a resolución completa.
//   2. Imagen reducida a la entrada del modelo → L → gris reconstruido vía
//      Lab (L,0,0) → RGB 0..1 → DDColor (sin normalización extra, el modelo
//      oficial usa do_normalize=False) → croma AB.
//   3. AB reescalado (bilineal, float) al tamaño original; L original + AB
//      → Lab → RGB → PNG.
// La luminancia nunca baja de resolución: solo la croma pasa por el modelo.
// La carga del modelo usa ModelLoader (js/model-loader.js): caché en memoria y
// Cache Storage (disco del usuario), ./models/ servido por la web o, si no está,
// descarga desde una URL remota de respaldo. Sesión en caché y fallback WebGPU
// → WASM también ante fallos en la primera inferencia.

const MODEL_URL = './models/ddcolor.onnx';
// Respaldo remoto (HuggingFace): ddcolor.onnx es un exporte propio
// (ver models/README.md) alojado en un repo del proyecto para que la
// colorización funcione en despliegues estáticos (Azure Static Web Apps).
const MODEL_REMOTE_URL = 'https://huggingface.co/RainBowDashX/photorestore-ddcolor/resolve/main/ddcolor.onnx';
const MODEL_MIN_BYTES = 100_000_000; // real: 270.255.132 bytes (fp32)
const INPUT_SIZE = 512;              // entrada fija del export: [1,3,512,512]

let session = null;
let backend = null;
let forceWasm = false;

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

// ---------------- Conversión sRGB <-> CIE Lab (fórmulas de OpenCV) ----------------

const srgbLineal = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const srgbGamma = c => c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
const labF = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
const labFInv = t => t * t * t > 0.008856 ? t * t * t : (t - 16 / 116) / 7.787;

// r,g,b en 0..255 -> [L, a, b] (L 0..100; a,b ~[-127,127])
export function rgbToLab(r, g, b) {
    const rl = srgbLineal(r / 255), gl = srgbLineal(g / 255), bl = srgbLineal(b / 255);
    const x = (0.412453 * rl + 0.357580 * gl + 0.180423 * bl) / 0.950456;
    const y = 0.212671 * rl + 0.715160 * gl + 0.072169 * bl;
    const z = (0.019334 * rl + 0.119193 * gl + 0.950227 * bl) / 1.088754;
    const fx = labF(x), fy = labF(y), fz = labF(z);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// [L, a, b] -> [r, g, b] en 0..255 con saturación
export function labToRgb(L, a, b) {
    const fy = (L + 16) / 116;
    const fx = fy + a / 500;
    const fz = fy - b / 200;
    const x = 0.950456 * labFInv(fx);
    const y = labFInv(fy);
    const z = 1.088754 * labFInv(fz);
    const rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    const gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    const bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    const c = v => {
        v = Math.round(srgbGamma(Math.max(0, Math.min(1, v))) * 255);
        return v < 0 ? 0 : v > 255 ? 255 : v;
    };
    return [c(rl), c(gl), c(bl)];
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

// ---------------- Punto de entrada ----------------

// Coloriza una foto en blanco y negro manteniendo su tamaño original.
// progressHelper (DotNetObjectReference): OnEstado(string) para la fase.
// Devuelve { resultUrl, backend, width, height }.
export async function colorize(bytes, mimeType, progressHelper) {
    try {
        return await colorizeInterno(bytes, mimeType, progressHelper);
    } catch (err) {
        // La sesión WebGPU puede crearse bien pero fallar en la primera
        // inferencia: se libera y se reintenta todo con WASM (CPU).
        if (forceWasm || backend !== 'webgpu') throw err;
        console.warn('WebGPU falló durante la inferencia; reintentando con WASM (CPU).', err);
        forceWasm = true;
        try { await session?.release?.(); } catch { /* mejor esfuerzo */ }
        session = null;
        backend = null;
        try {
            return await colorizeInterno(bytes, mimeType, progressHelper);
        } catch (errWasm) {
            throw new Error(`Falló WebGPU (${err.message}) y también WASM (${errWasm.message})`);
        }
    }
}

async function colorizeInterno(bytes, mimeType, progressHelper) {
    const onProgreso = progressHelper
        ? p => progressHelper.invokeMethodAsync('OnEstado', p < 0
            ? 'Descargando modelo de colorización (solo la primera vez)…'
            : `Descargando modelo de colorización (solo la primera vez)… ${p}%`)
        : null;
    const sess = await getSession(onProgreso);

    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const w = bitmap.width, h = bitmap.height;

    // L de la imagen original a resolución completa.
    const src = new OffscreenCanvas(w, h);
    const srcCtx = src.getContext('2d', { willReadFrequently: true });
    srcCtx.drawImage(bitmap, 0, 0);
    const imgData = srcCtx.getImageData(0, 0, w, h).data;
    const n = w * h;
    const origL = new Float32Array(n);
    for (let i = 0; i < n; i++)
        origL[i] = rgbToLab(imgData[i * 4], imgData[i * 4 + 1], imgData[i * 4 + 2])[0];

    // Entrada del modelo: gris reconstruido vía Lab al tamaño de entrada
    // (export propio fp32, tamaño fijo verificado: [1,3,512,512]).
    const S = INPUT_SIZE;
    const small = new OffscreenCanvas(S, S);
    const smallCtx = small.getContext('2d', { willReadFrequently: true });
    smallCtx.imageSmoothingEnabled = true;
    smallCtx.imageSmoothingQuality = 'high';
    smallCtx.drawImage(bitmap, 0, 0, S, S);
    bitmap.close();
    const smallData = smallCtx.getImageData(0, 0, S, S).data;

    const ns = S * S;
    const chw = new Float32Array(3 * ns);
    for (let i = 0; i < ns; i++) {
        const L = rgbToLab(smallData[i * 4], smallData[i * 4 + 1], smallData[i * 4 + 2])[0];
        const [r, g, b] = labToRgb(L, 0, 0); // gris vía Lab, como el pipeline oficial
        chw[i] = r / 255;
        chw[ns + i] = g / 255;
        chw[2 * ns + i] = b / 255;
    }

    if (progressHelper)
        await progressHelper.invokeMethodAsync('OnEstado', 'Colorizando…');
    const res = await sess.run({
        [sess.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, S, S])
    });
    const ab = res[sess.outputNames[0]].data; // [1, 2, S, S]

    // AB a resolución original (bilineal float) y fusión con la L original.
    const aFull = resizeBilinear(ab.subarray(0, ns), S, S, w, h);
    const bFull = resizeBilinear(ab.subarray(ns, 2 * ns), S, S, w, h);

    const out = new OffscreenCanvas(w, h);
    const outCtx = out.getContext('2d');
    const outImg = new ImageData(w, h);
    for (let i = 0; i < n; i++) {
        const [r, g, b] = labToRgb(origL[i], aFull[i], bFull[i]);
        outImg.data[i * 4] = r;
        outImg.data[i * 4 + 1] = g;
        outImg.data[i * 4 + 2] = b;
        outImg.data[i * 4 + 3] = 255;
    }
    outCtx.putImageData(outImg, 0, 0);

    const blob = await out.convertToBlob({ type: 'image/png' });
    return {
        resultUrl: URL.createObjectURL(blob),
        backend,
        width: w,
        height: h
    };
}
