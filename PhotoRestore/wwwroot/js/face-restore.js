// face-restore.js — Restauración de caras con GFPGAN v1.4 en el navegador.
// Pipeline: detección de caras (SCRFD 2.5G, con 5 puntos faciales) → alineado
// de cada cara a 512×512 (transformación de semejanza a la plantilla ArcFace)
// → restauración con GFPGAN → pegado en la foto original con máscara suave.
// Mismo patrón que upscaler.js: descarga manual del modelo con validación y un
// reintento, sesiones en caché y fallback explícito WebGPU → WASM.
//
// Fallback robusto: la creación de sesión puede ir bien en WebGPU pero fallar
// un op en la PRIMERA inferencia (p. ej. shape computation no soportada). Por
// eso restoreFaces reintenta TODA la pipeline con WASM forzado si cualquier
// sesión WebGPU falla en tiempo de ejecución; los bytes del modelo se cachean
// para no repetir la descarga (340 MB).

const DETECTOR_URL = './models/scrfd-2.5g.onnx';
const GFPGAN_URL = './models/gfpgan-v1.4.onnx';
const DETECTOR_MIN_BYTES = 1_000_000;   // real: 3.291.737 bytes
const GFPGAN_MIN_BYTES = 100_000_000;   // real: ~340 MB (fp32)

const DET_SIZE = 640;   // entrada del detector SCRFD
const FACE_SIZE = 512;  // entrada/salida de GFPGAN
const SCORE_MIN = 0.5;  // confianza mínima de detección
const NMS_IOU = 0.4;
const MAX_CARAS = 8;

// Plantilla ArcFace de 5 puntos faciales (112×112) escalada ×4 a 512×512,
// la misma que usan facexlib/GFPGAN para alinear.
const PLANTILLA_512 = [
    [38.2946 * 4, 51.6963 * 4],
    [73.5318 * 4, 51.5014 * 4],
    [56.0252 * 4, 71.7366 * 4],
    [41.5493 * 4, 92.3655 * 4],
    [70.7299 * 4, 92.2041 * 4]
];

const sessions = new Map();   // url -> InferenceSession
const backends = new Map();   // url -> 'webgpu' | 'wasm'
const modelCache = new Map(); // url -> Uint8Array (evita redescargar al reintentar)
let forceWasm = false;        // se activa si WebGPU falla en la primera inferencia

function webGpuSupported() {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

// Descarga un modelo con un reintento y mensajes de error autoexplicativos.
async function fetchModel(url, minBytes) {
    if (modelCache.has(url)) return modelCache.get(url);
    let lastError = null;
    for (let intento = 1; intento <= 2; intento++) {
        try {
            console.log(`Descargando ${url} (intento ${intento})…`);
            const resp = await fetch(url, { cache: 'no-cache' });
            if (!resp.ok)
                throw new Error(`Error del servidor al descargar el modelo: HTTP ${resp.status}`);
            const buffer = await resp.arrayBuffer();
            if (buffer.byteLength < minBytes)
                throw new Error(`El modelo descargado parece incompleto (${buffer.byteLength} bytes); recarga la página e inténtalo de nuevo.`);
            const bytes = new Uint8Array(buffer);
            modelCache.set(url, bytes);
            return bytes;
        } catch (err) {
            lastError = err;
            console.warn(`Fallo al descargar ${url} (intento ${intento}).`, err);
            if (intento < 2)
                await new Promise(r => setTimeout(r, 800));
        }
    }
    if (lastError instanceof TypeError) // error de red de fetch
        throw new Error(`No se pudo descargar el modelo (¿el servidor sigue corriendo?): ${lastError.message}`);
    throw lastError;
}

// Sesión en caché por modelo, creada desde bytes; WebGPU con caída a WASM (CPU).
async function getSession(url, minBytes) {
    if (sessions.has(url)) return sessions.get(url);
    const modelBytes = await fetchModel(url, minBytes);
    const providers = (!forceWasm && webGpuSupported()) ? ['webgpu', 'wasm'] : ['wasm'];
    let lastError = null;
    for (const ep of providers) {
        try {
            const sess = await ort.InferenceSession.create(modelBytes, {
                executionProviders: [ep],
                graphOptimizationLevel: 'all'
            });
            sessions.set(url, sess);
            backends.set(url, ep);
            return sess;
        } catch (err) {
            lastError = err;
        }
    }
    throw lastError;
}

// Libera las sesiones cacheadas (se usa antes de reintentar con WASM).
async function liberarSesiones() {
    for (const sess of sessions.values()) {
        try { await sess.release?.(); } catch { /* mejor esfuerzo */ }
    }
    sessions.clear();
    backends.clear();
}

// ---------------- Detección (SCRFD 2.5G bnkps) ----------------

async function detectarCaras(sess, srcCanvas, w, h) {
    // Letterbox a 640×640: escala uniforme, anclado arriba-izquierda, relleno 0.
    const scale = Math.min(DET_SIZE / w, DET_SIZE / h);
    const nw = Math.max(1, Math.round(w * scale));
    const nh = Math.max(1, Math.round(h * scale));
    const canvas = new OffscreenCanvas(DET_SIZE, DET_SIZE);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(srcCanvas, 0, 0, nw, nh);
    const img = ctx.getImageData(0, 0, DET_SIZE, DET_SIZE);

    // SCRFD (insightface) espera BGR normalizado a [-1, 1].
    const n = DET_SIZE * DET_SIZE;
    const d = img.data;
    const chw = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        chw[i]         = (d[i * 4 + 2] - 127.5) / 127.5;
        chw[n + i]     = (d[i * 4 + 1] - 127.5) / 127.5;
        chw[2 * n + i] = (d[i * 4]     - 127.5) / 127.5;
    }
    const res = await sess.run({
        [sess.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, DET_SIZE, DET_SIZE])
    });

    // Salidas por stride (8/16/32): scores ya son probabilidades (el grafo
    // incluye los Sigmoid), distancias de caja y offsets de 5 puntos, con
    // 2 anclas por celda (índice = celda*2 + ancla).
    const candidatos = [];
    for (const stride of [8, 16, 32]) {
        const scores = res[`score_${stride}`].data;
        const bboxes = res[`bbox_${stride}`].data;
        const kps = res[`kps_${stride}`].data;
        const fmc = DET_SIZE / stride;
        for (let idx = 0; idx < scores.length; idx++) {
            const score = scores[idx];
            if (score < SCORE_MIN) continue;
            const celda = Math.floor(idx / 2);
            const cx = celda % fmc, cy = Math.floor(celda / fmc);
            const b = idx * 4;
            const puntos = [];
            for (let k = 0; k < 5; k++) {
                puntos.push([(cx + kps[idx * 10 + k * 2]) * stride,
                             (cy + kps[idx * 10 + k * 2 + 1]) * stride]);
            }
            candidatos.push({
                score,
                x1: (cx - bboxes[b]) * stride,
                y1: (cy - bboxes[b + 1]) * stride,
                x2: (cx + bboxes[b + 2]) * stride,
                y2: (cy + bboxes[b + 3]) * stride,
                puntos
            });
        }
    }

    const sel = nms(candidatos, NMS_IOU).slice(0, MAX_CARAS);
    for (const c of sel) { // deshacer el letterbox
        c.x1 /= scale; c.y1 /= scale; c.x2 /= scale; c.y2 /= scale;
        c.puntos = c.puntos.map(([x, y]) => [x / scale, y / scale]);
    }
    return sel;
}

function iou(a, b) {
    const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
    const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    const area = c => Math.max(0, c.x2 - c.x1) * Math.max(0, c.y2 - c.y1);
    return inter / (area(a) + area(b) - inter || 1);
}

function nms(candidatos, iouMax) {
    const orden = [...candidatos].sort((a, b) => b.score - a.score);
    const keep = [];
    for (const c of orden)
        if (keep.every(k => iou(k, c) < iouMax))
            keep.push(c);
    return keep;
}

// ---------------- Alineado (semejanza 2D, Umeyama sin reflexión) ----------------

// Ajuste por mínimos cuadrados de w = a·z + b (a complejo: escala+rotación).
// Devuelve [a, b, c, d, e, f] para canvas.setTransform:
//   x' = a·x + c·y + e ;  y' = b·x + d·y + f
function semejanza(src, dst) {
    const n = src.length;
    let sx = 0, sy = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        sx += src[i][0]; sy += src[i][1];
        dx += dst[i][0]; dy += dst[i][1];
    }
    sx /= n; sy /= n; dx /= n; dy /= n;
    let numRe = 0, numIm = 0, den = 0;
    for (let i = 0; i < n; i++) {
        const zx = src[i][0] - sx, zy = src[i][1] - sy;
        const wx = dst[i][0] - dx, wy = dst[i][1] - dy;
        numRe += zx * wx + zy * wy;   // Re(conj(z)·w)
        numIm += zx * wy - zy * wx;   // Im(conj(z)·w)
        den += zx * zx + zy * zy;
    }
    const ar = numRe / den, ai = numIm / den;
    const br = dx - (ar * sx - ai * sy);
    const bi = dy - (ar * sy + ai * sx);
    return [ar, ai, -ai, ar, br, bi];
}

function invertirAffine([a, b, c, d, e, f]) {
    const det = a * d - b * c;
    return [d / det, -b / det, -c / det, a / det,
            (c * f - d * e) / det, (b * e - a * f) / det];
}

// ---------------- GFPGAN ----------------

const clamp255 = v => {
    v = Math.round(v * 255);
    return v < 0 ? 0 : v > 255 ? 255 : v;
};

// Alinea la cara a 512×512, la restaura con GFPGAN y devuelve el canvas
// restaurado (con máscara suave aplicada) y la matriz inversa para pegarlo.
async function restaurarCara(sess, srcCanvas, puntos) {
    const M = semejanza(puntos, PLANTILLA_512);

    const cara = new OffscreenCanvas(FACE_SIZE, FACE_SIZE);
    const ctx = cara.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.setTransform(...M);
    ctx.drawImage(srcCanvas, 0, 0);
    const img = ctx.getImageData(0, 0, FACE_SIZE, FACE_SIZE);

    // GFPGAN espera RGB normalizado a [-1, 1].
    const n = FACE_SIZE * FACE_SIZE;
    const d = img.data;
    const chw = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        chw[i]         = d[i * 4]     / 127.5 - 1;
        chw[n + i]     = d[i * 4 + 1] / 127.5 - 1;
        chw[2 * n + i] = d[i * 4 + 2] / 127.5 - 1;
    }
    const res = await sess.run({
        [sess.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, FACE_SIZE, FACE_SIZE])
    });
    const out = res[sess.outputNames[0]].data; // [-1, 1]

    const restaurada = new OffscreenCanvas(FACE_SIZE, FACE_SIZE);
    const ctxOut = restaurada.getContext('2d');
    const outImg = new ImageData(FACE_SIZE, FACE_SIZE);
    for (let i = 0; i < n; i++) {
        outImg.data[i * 4]     = clamp255((out[i] + 1) / 2);
        outImg.data[i * 4 + 1] = clamp255((out[n + i] + 1) / 2);
        outImg.data[i * 4 + 2] = clamp255((out[2 * n + i] + 1) / 2);
        outImg.data[i * 4 + 3] = 255;
    }
    ctxOut.putImageData(outImg, 0, 0);
    aplicarMascaraSuave(restaurada);
    return { cara: restaurada, Minv: invertirAffine(M) };
}

// Recorta la cara restaurada con una elipse de bordes difuminados (degradado
// radial en el canal alfa) centrada en la zona facial de la plantilla 512.
function aplicarMascaraSuave(canvas) {
    const mask = new OffscreenCanvas(FACE_SIZE, FACE_SIZE);
    const m = mask.getContext('2d');
    m.translate(230, 285);
    m.scale(1, 1.2); // elipse vertical
    const g = m.createRadialGradient(0, 0, 120, 0, 0, 185);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.8, 'rgba(255,255,255,1)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    m.fillStyle = g;
    m.fillRect(-300, -300, 600, 600);

    const ctx = canvas.getContext('2d');
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(mask, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
}

// ---------------- Punto de entrada ----------------

// Restaura las caras de la foto a su tamaño original.
// progressHelper (DotNetObjectReference): OnEstado(string) para la fase y
// OnCaraDone(int done, int total) tras cada cara.
// Devuelve { resultUrl, backend, width, height, faces }; resultUrl es null
// si no se detectó ninguna cara.
export async function restoreFaces(bytes, mimeType, progressHelper) {
    try {
        return await restoreFacesInterno(bytes, mimeType, progressHelper);
    } catch (err) {
        // La sesión WebGPU puede crearse bien pero fallar en la primera
        // inferencia: se libera todo y se reintenta la pipeline con WASM.
        const webgpuEnUso = backends.get(DETECTOR_URL) === 'webgpu' ||
                            backends.get(GFPGAN_URL) === 'webgpu';
        if (forceWasm || !webgpuEnUso) throw err;
        console.warn('WebGPU falló durante la inferencia; reintentando con WASM (CPU).', err);
        forceWasm = true;
        await liberarSesiones();
        try {
            return await restoreFacesInterno(bytes, mimeType, progressHelper);
        } catch (errWasm) {
            throw new Error(`Falló WebGPU (${err.message}) y también WASM (${errWasm.message})`);
        }
    }
}

async function restoreFacesInterno(bytes, mimeType, progressHelper) {
    const detSess = await getSession(DETECTOR_URL, DETECTOR_MIN_BYTES);
    const ganSess = await getSession(GFPGAN_URL, GFPGAN_MIN_BYTES);

    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const w = bitmap.width, h = bitmap.height;
    const src = new OffscreenCanvas(w, h);
    const srcCtx = src.getContext('2d', { willReadFrequently: true });
    srcCtx.drawImage(bitmap, 0, 0);
    bitmap.close();

    if (progressHelper)
        await progressHelper.invokeMethodAsync('OnEstado', 'Analizando caras…');
    const caras = await detectarCaras(detSess, src, w, h);

    const dst = new OffscreenCanvas(w, h);
    const dstCtx = dst.getContext('2d');
    dstCtx.drawImage(src, 0, 0);

    if (caras.length === 0)
        return { resultUrl: null, backend: backends.get(GFPGAN_URL), width: w, height: h, faces: 0 };

    dstCtx.imageSmoothingEnabled = true;
    dstCtx.imageSmoothingQuality = 'high';
    for (let i = 0; i < caras.length; i++) {
        if (progressHelper)
            await progressHelper.invokeMethodAsync('OnCaraDone', i + 1, caras.length);
        const { cara, Minv } = await restaurarCara(ganSess, src, caras[i].puntos);
        dstCtx.setTransform(...Minv);
        dstCtx.drawImage(cara, 0, 0);
        dstCtx.setTransform(1, 0, 0, 1, 0, 0);
    }

    const blob = await dst.convertToBlob({ type: 'image/png' });
    return {
        resultUrl: URL.createObjectURL(blob),
        backend: backends.get(GFPGAN_URL),
        width: w,
        height: h,
        faces: caras.length
    };
}
