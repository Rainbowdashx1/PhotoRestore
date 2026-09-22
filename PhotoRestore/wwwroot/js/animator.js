// animator.js — Animación de imágenes con ffmpeg.wasm para PhotoRestore.
// ffmpeg.wasm se importa dinámicamente desde jsDelivr (prototipo, versión
// fijada) y su núcleo WASM (~31 MB) solo se descarga la primera vez que se
// genera una animación. Se usa el core single-thread (@ffmpeg/core), que no
// requiere SharedArrayBuffer ni cabeceras COOP/COEP: válido en hosting
// estático. Para producción, autoalojar los dist en wwwroot/lib/.

const FFMPEG_ESM = 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/+esm';
// Worker interno de ffmpeg ya empaquetado por jsDelivr (autocontenido): los
// workers solo pueden crearse desde URLs same-origin o blob, así que el
// archivo original del CDN no sirve directamente.
const WORKER_ESM = 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm/worker.js/+esm';
const CORE_BASE = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/esm';
const MAX_LADO = 640;   // lado máximo del lienzo, para acotar el tamaño del GIF
const FPS = 24;

let ffmpeg = null;
let ffmpegCargando = null; // promesa en curso, para no cargar dos veces

// Helpers de vista previa (mismo patrón que upscaler.js).

export function createObjectUrl(bytes, mimeType) {
    return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

export function revokeObjectUrl(url) {
    if (url) URL.revokeObjectURL(url);
}

// Dimensiones naturales de la imagen seleccionada.
export async function getImageSize(bytes, mimeType) {
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return size;
}

// Carga perezosa del núcleo de ffmpeg.wasm. onEstado es un callback opcional
// con el texto de estado (para la tarjeta de progreso de Blazor).
async function cargarFFmpeg(onEstado) {
    if (ffmpeg) return;
    if (!ffmpegCargando) {
        ffmpegCargando = (async () => {
            onEstado?.('Descargando ffmpeg.wasm (~31 MB, solo la primera vez)…');
            const { FFmpeg } = await import(/* @vite-ignore */ FFMPEG_ESM);
            const instancia = new FFmpeg();
            // toBlobURL: tanto el core como el worker exigen URLs
            // same-origin o blob.
            const [coreURL, wasmURL, classWorkerURL] = await Promise.all([
                toBlobUrl(`${CORE_BASE}/ffmpeg-core.js`, 'text/javascript'),
                toBlobUrl(`${CORE_BASE}/ffmpeg-core.wasm`, 'application/wasm'),
                toBlobUrl(WORKER_ESM, 'text/javascript')
            ]);
            onEstado?.('Iniciando ffmpeg…');
            await instancia.load({ coreURL, wasmURL, classWorkerURL });
            ffmpeg = instancia;
        })().catch(err => {
            ffmpegCargando = null; // permite reintentar tras un fallo de red
            throw err;
        });
    }
    await ffmpegCargando;
}

async function toBlobUrl(url, mimeType) {
    const resp = await fetch(url);
    if (!resp.ok)
        throw new Error(`No se pudo descargar ${url}: HTTP ${resp.status}`);
    return URL.createObjectURL(new Blob([await resp.arrayBuffer()], { type: mimeType }));
}

// Decodifica la imagen y la reescala a un lienzo de lado máx. MAX_LADO con
// dimensiones pares (libx264 exige ancho y alto pares). Devuelve el PNG
// resultante y sus dimensiones.
async function prepararImagen(bytes, mimeType) {
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeType }));
    const escala = Math.min(1, MAX_LADO / Math.max(bitmap.width, bitmap.height));
    let w = Math.round(bitmap.width * escala);
    let h = Math.round(bitmap.height * escala);
    w -= w % 2;
    h -= h % 2;
    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, w, h);
    bitmap.close();
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return { png: new Uint8Array(await blob.arrayBuffer()), width: w, height: h };
}

// Cadena de filtros de ffmpeg para cada tipo de animación. D = duración (s),
// bg = fondo ('white' | 'black' | 'none' = transparente), W/H = tamaño del
// lienzo. La transparencia solo se conserva en GIF (MP4/yuv420p no tiene
// canal alfa). Devuelve { entradas: [...args de entrada...], cadena: 'filtro' }
// donde la cadena opera sobre las entradas y produce un único stream de vídeo.
function construirFiltro(tipo, dur, bg, W, H) {
    const D = dur;
    const transparente = bg === 'none';
    // 'black@0' = color con alfa 0, válido en todos los filtros de dibujo.
    const relleno = transparente ? 'black@0' : bg;
    // Con fondo de color la imagen se compone SIEMPRE sobre un lienzo sólido
    // (si el PNG trae transparencia propia, sin lienzo esas zonas quedarían
    // transparentes en el GIF final en vez del color elegido).
    // Con fondo transparente, format=rgba debe ir dentro del grafo lavfi: si
    // no, la fuente color negocia un formato sin alfa y el @0 se pierde
    // (el fondo sale negro opaco).
    const lienzoSrc = `color=c=${relleno}:s=${W}x${H}:r=${FPS}` + (transparente ? ',format=rgba' : '');
    const entradasConLienzo = [
        '-f', 'lavfi', '-t', String(D), '-i', lienzoSrc,
        '-loop', '1', '-t', String(D), '-i', 'input.png'
    ];
    const entradaSola = ['-loop', '1', '-t', String(D), '-i', 'input.png'];
    switch (tipo) {
        // En coordenadas de vídeo (eje Y hacia abajo) el signo positivo gira
        // en sentido horario. ow/oh fijos (diagonal del lienzo) porque ffmpeg
        // no admite tamaños de fotograma variables. El bucle es perfecto:
        // en t=D la imagen ha dado exactamente una vuelta.
        case 'rotate-cw':
        case 'rotate-ccw': {
            const diag = Math.ceil(Math.hypot(W, H));
            const lado = diag + (diag % 2);
            const signo = tipo === 'rotate-cw' ? '' : '-';
            const giro = `rotate=a=${signo}2*PI*t/${D}:ow=${lado}:oh=${lado}:c=${relleno},fps=${FPS}`;
            return transparente
                ? { entradas: entradaSola, cadena: `[0:v]format=rgba,${giro}` }
                : { entradas: entradasConLienzo, cadena: `[0:v][1:v]overlay=(W-w)/2:(H-h)/2,${giro}` };
        }
        // La imagen entra desde fuera del borde izquierdo hasta quedar
        // centrada; min() la detiene al final.
        case 'slide':
            return {
                entradas: entradasConLienzo,
                cadena: `[0:v][1:v]overlay=x=-w+((W-w)/2+w)*min(t/${D}\\,1):y=(H-h)/2,fps=${FPS}`
            };
        // Fundido de aparición y desaparición. Con fondo de color se funde el
        // fotograma completo (lienzo + imagen) hacia ese color; con fondo
        // transparente se funde el canal alfa de la imagen (en GIF la
        // transparencia es binaria, el fundido se ve escalonado).
        case 'fade': {
            const F = Math.min(1, D / 2);
            return transparente
                ? {
                    entradas: entradaSola,
                    cadena: `[0:v]format=rgba,fade=t=in:st=0:d=${F}:alpha=1,fade=t=out:st=${D - F}:d=${F}:alpha=1,fps=${FPS}`
                }
                : {
                    entradas: entradasConLienzo,
                    cadena: `[0:v][1:v]overlay=(W-w)/2:(H-h)/2,fade=t=in:st=0:d=${F}:color=${bg},fade=t=out:st=${D - F}:d=${F}:color=${bg},fps=${FPS}`
                };
        }
        // Revelado tipo "carga": una máscara animada (bloque blanco sobre
        // fondo negro) se aplica como alfa a la imagen. El bloque entra desde
        // abajo (y: h → 0 → h, onda triangular en D segundos), así que la zona
        // visible crece de abajo hacia arriba, se completa y retrocede, en
        // bucle perfecto.
        // OJO: alphamerge REEMPLAZA el alfa, así que la máscara se multiplica
        // primero por el alfa original del PNG (blend=multiply) para no
        // perder su transparencia propia (si no, esas zonas saldrían negras).
        case 'wipe': {
            const onda = `h*abs(2*mod(t\\,${D})/${D}-1)`;
            const mascara =
                `[0:v][2:v]overlay=x=0:y=${onda},format=gray[m];` +
                `[1:v]format=rgba,split[i][j];` +
                `[j]alphaextract[a];` +
                `[m][a]blend=all_mode=multiply[ma];` +
                `[i][ma]alphamerge`;
            const entradas = [
                '-f', 'lavfi', '-t', String(D), '-i', `color=c=black:s=${W}x${H}:r=${FPS}`, // [0] base de la máscara
                '-loop', '1', '-t', String(D), '-i', 'input.png',                          // [1] imagen
                '-f', 'lavfi', '-t', String(D), '-i', `color=c=white:s=${W}x${H}:r=${FPS}` // [2] bloque de la máscara
            ];
            if (transparente)
                return { entradas, cadena: `${mascara},fps=${FPS}` };
            // Con fondo de color, la imagen con alfa se compone sobre el lienzo.
            return {
                entradas: [
                    ...entradas,
                    '-f', 'lavfi', '-t', String(D), '-i', lienzoSrc // [3] lienzo de fondo
                ],
                cadena: `${mascara}[ia];[3:v][ia]overlay=(W-w)/2:(H-h)/2,fps=${FPS}`
            };
        }
        default:
            throw new Error(`Tipo de animación no soportado: ${tipo}`);
    }
}

// Genera la animación y devuelve { resultUrl, width, height, format }.
// tipo: 'rotate-cw' | 'rotate-ccw' | 'slide' | 'fade' | 'wipe'
// dur: 2 | 3 | 5   bg: 'white' | 'black' | 'none'
// formato: 'gif' | 'apng' | 'mp4'
//   - GIF: transparencia binaria (el fundido transparente se ve escalonado).
//   - APNG: alfa completo de 8 bits → fundido transparente suave; se muestra
//     como <img> y hace bucle solo (-plays 0).
//   - MP4: sin canal alfa (yuv420p).
// progressHelper: DotNetObjectReference con OnProgreso(int) y OnEstado(string).
export async function animate(bytes, mimeType, tipo, dur, bg, formato, progressHelper) {
    const onEstado = t => progressHelper?.invokeMethodAsync('OnEstado', t);
    const onProgreso = p => progressHelper?.invokeMethodAsync('OnProgreso', p);

    await cargarFFmpeg(onEstado);
    onEstado?.('Preparando fotogramas…');

    const { png, width, height } = await prepararImagen(bytes, mimeType);
    const { entradas, cadena } = construirFiltro(tipo, dur, bg, width, height);

    const onProgress = ({ progress }) => onProgreso(Math.round(Math.min(1, Math.max(0, progress)) * 100));
    ffmpeg.on('progress', onProgress);

    // Últimas líneas del log de ffmpeg, para que los errores digan la causa.
    const logs = [];
    const onLog = ({ message }) => {
        logs.push(message);
        if (logs.length > 25) logs.shift();
    };
    ffmpeg.on('log', onLog);

    // exec() devuelve el código de salida; 0 = éxito.
    const ejecutar = async (args, fase) => {
        const code = await ffmpeg.exec(args);
        if (code !== 0)
            throw new Error(`ffmpeg falló en «${fase}» (código ${code}): ${logs.slice(-4).join(' | ')}`);
    };

    try {
        await ffmpeg.writeFile('input.png', png);

        let datos, mimeSalida;
        if (formato === 'gif') {
            // Dos pasadas: primero una paleta óptima, luego el GIF con ella.
            onEstado?.('Generando paleta de colores…');
            await ejecutar([...entradas, '-filter_complex', `${cadena},palettegen`, '-y', 'palette.png'], 'paleta');
            onEstado?.('Codificando GIF…');
            // La paleta entra como última entrada: su índice es el nº de -i.
            const paletaIdx = entradas.filter(a => a === '-i').length;
            await ejecutar([
                ...entradas, '-i', 'palette.png',
                '-filter_complex', `${cadena}[x];[x][${paletaIdx}:v]paletteuse`,
                '-loop', '0', '-y', 'salida.gif'
            ], 'GIF');
            datos = await ffmpeg.readFile('salida.gif');
            mimeSalida = 'image/gif';
        } else if (formato === 'apng') {
            // APNG conserva el alfa completo: se fuerza format=rgba (el
            // encoder no acepta yuv420p) y -f apng (con la extensión .png
            // ffmpeg asumiría el muxer image2 de PNG suelto).
            onEstado?.('Codificando APNG…');
            await ejecutar([
                ...entradas,
                '-filter_complex', `${cadena},format=rgba`,
                '-c:v', 'apng', '-f', 'apng', '-plays', '0', '-y', 'salida.png'
            ], 'APNG');
            datos = await ffmpeg.readFile('salida.png');
            mimeSalida = 'image/apng';
        } else {
            onEstado?.('Codificando MP4…');
            await ejecutar([
                ...entradas,
                '-filter_complex', `${cadena},format=yuv420p`,
                '-c:v', 'libx264', '-movflags', '+faststart', '-y', 'salida.mp4'
            ], 'MP4');
            datos = await ffmpeg.readFile('salida.mp4');
            mimeSalida = 'video/mp4';
        }

        return {
            resultUrl: URL.createObjectURL(new Blob([datos], { type: mimeSalida })),
            width, height, format: formato
        };
    } finally {
        ffmpeg.off('progress', onProgress);
        ffmpeg.off('log', onLog);
        // Limpieza del FS virtual; ignoramos errores si un archivo no existe.
        for (const f of ['input.png', 'palette.png', 'salida.gif', 'salida.png', 'salida.mp4'])
            try { await ffmpeg.deleteFile(f); } catch { /* no existía */ }
    }
}
