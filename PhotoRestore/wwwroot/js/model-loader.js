// model-loader.js — carga de modelos ONNX con caché persistente en el PC del
// usuario. Los modelos NO se descargan en cada visita:
//
//   1. Memoria (misma sesión de la página).
//   2. Cache Storage del navegador (disco del usuario; persiste entre visitas).
//   3. ./models/… servido por la propia web (desarrollo local; en el repo solo
//      están los modelos pequeños).
//   4. URL remota de respaldo (HuggingFace) para despliegues estáticos sin los
//      .onnx grandes en el servidor (p. ej. Azure Static Web Apps, donde el
//      repo no puede incluir archivos de cientos de MB).
//
// Lo obtenido de 3 o 4 se guarda en Cache Storage, de modo que la descarga
// grande solo ocurre una vez por navegador.
//
// Es un script clásico (no módulo) a propósito: se carga con <script src> en
// index.html y expone window.ModelLoader, sin depender de imports entre
// módulos (fingerprinting de assets estáticos).
window.ModelLoader = (() => {
    const CACHE_NAME = 'photorestore-models-v1';
    const memoria = new Map();   // url -> Uint8Array

    function cacheStorageDisponible() {
        try { return typeof caches !== 'undefined'; } catch { return false; }
    }

    async function leerDeCacheStorage(url, minBytes) {
        if (!cacheStorageDisponible()) return null;
        try {
            const resp = await (await caches.open(CACHE_NAME)).match(url);
            if (!resp) return null;
            const bytes = new Uint8Array(await resp.arrayBuffer());
            // Por si quedó una descarga incompleta de una versión anterior.
            if (bytes.byteLength < minBytes) return null;
            return bytes;
        } catch {
            return null;   // almacenamiento deshabilitado o lleno: se descarga
        }
    }

    function guardarEnCacheStorage(url, bytes) {
        if (!cacheStorageDisponible()) return;
        caches.open(CACHE_NAME)
            .then(c => c.put(url, new Response(bytes, {
                headers: { 'Content-Type': 'application/octet-stream' }
            })))
            .catch(err => console.warn(`No se pudo guardar ${url} en Cache Storage (¿cuota llena?).`, err));
    }

    // Descarga con progreso. onProgreso(porcentaje) con 0..100, o -1 si el
    // servidor no informa del tamaño total. Se llama como mucho una vez por
    // punto porcentual.
    async function descargar(url, minBytes, onProgreso) {
        const resp = await fetch(url);
        if (!resp.ok)
            throw new Error(`Error al descargar el modelo (${url}): HTTP ${resp.status}`);
        const total = Number(resp.headers.get('content-length')) || 0;
        let bytes;
        if (resp.body && onProgreso) {
            const reader = resp.body.getReader();
            const chunks = [];
            let recibidos = 0, ultimoPct = -2;
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                recibidos += value.byteLength;
                const pct = total ? Math.floor(recibidos * 100 / total) : -1;
                if (pct !== ultimoPct) {
                    ultimoPct = pct;
                    await onProgreso(pct);
                }
            }
            bytes = new Uint8Array(recibidos);
            let off = 0;
            for (const c of chunks) { bytes.set(c, off); off += c.byteLength; }
        } else {
            bytes = new Uint8Array(await resp.arrayBuffer());
        }
        if (bytes.byteLength < minBytes)
            throw new Error(`El modelo descargado parece incompleto (${bytes.byteLength} bytes); recarga la página e inténtalo de nuevo.`);
        return bytes;
    }

    // Carga un modelo aplicando la cadena de fuentes; un reintento si falla.
    // remoteUrl puede ser null (modelo solo disponible servido por la web).
    async function cargarModelo({ url, remoteUrl = null, minBytes = 0, onProgreso = null }) {
        if (memoria.has(url)) return memoria.get(url);

        const cacheados = await leerDeCacheStorage(url, minBytes);
        if (cacheados) {
            memoria.set(url, cacheados);
            return cacheados;
        }

        let lastError = null;
        for (let intento = 1; intento <= 2; intento++) {
            try {
                console.log(`Descargando ${url} (intento ${intento})…`);
                let bytes;
                try {
                    bytes = await descargar(url, minBytes, onProgreso);
                } catch (errLocal) {
                    if (!remoteUrl) throw errLocal;
                    console.warn(`${url} no está en este despliegue; se descarga desde ${remoteUrl}`, errLocal);
                    bytes = await descargar(remoteUrl, minBytes, onProgreso);
                }
                memoria.set(url, bytes);
                guardarEnCacheStorage(url, bytes);
                return bytes;
            } catch (err) {
                lastError = err;
                console.warn(`Fallo al descargar ${url} (intento ${intento}).`, err);
                if (intento < 2)
                    await new Promise(r => setTimeout(r, 800));
            }
        }
        if (lastError instanceof TypeError) // error de red de fetch
            throw new Error(`No se pudo descargar el modelo (revisa tu conexión a internet): ${lastError.message}`);
        throw lastError;
    }

    return { cargarModelo };
})();
