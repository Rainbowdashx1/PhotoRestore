# PhotoRestore

Herramienta web de restauración y retoque de imágenes con IA que corre **100% en el navegador** — la imagen del usuario nunca sale de su dispositivo. Blazor WebAssembly (.NET 10) para la UI y ONNX Runtime Web (WebGPU → WASM) para la inferencia.

> **Nombre en evaluación**: el proyecto ya no es solo restauración (también quita fondos). Candidato recomendado: **PhotoLab**; otras opciones discutidas: PixelFix, FotoTaller, Revela, ClearShot.

## Modos

El selector de modo son tarjetas con el nombre del modelo y una descripción breve de para qué sirve y cuándo usarlo (`.mode-card` en `app-custom.css`).

| Modo | Modelo | Qué hace | Tamaño modelo |
|---|---|---|---|
| **General** | Real-ESRGAN-General-x4v3 | Restaura y amplía (x1/x2/x4 a elección), con denoise incorporado | 4,9 MB |
| **Caras** | GFPGAN v1.4 + detector SCRFD 2.5G | Detecta caras, las alinea y restaura cada una a 512×512, las pega de vuelta con máscara difuminada | 340 MB + 3,3 MB |
| **Colorizar** | DDColor-tiny | Convierte fotos en blanco y negro a color, manteniendo el tamaño original | 258 MB |
| **Quitar fondo** | RMBG-1.4 | Detecta el sujeto principal automáticamente (sin selección manual) y devuelve PNG con transparencia, en el tamaño original | 176 MB |

Detalles de tensores, orígenes, licencias y parches de cada modelo: [`PhotoRestore/wwwroot/models/README.md`](PhotoRestore/wwwroot/models/README.md).

## Cómo ejecutar

```bash
cd PhotoRestore
dotnet run          # http://localhost:5249
```

O desde Visual Studio (perfil https: `https://localhost:7258`). Abrir `/` (la página de restauración es la home). Usar **Chrome o Edge** para soporte WebGPU.

> Si agregas o reemplazas archivos en `wwwroot/` mientras el servidor corre, reinícialo: el manifiesto de static web assets se genera al arrancar y los archivos nuevos dan 404 hasta reiniciar.

## Arquitectura

```
Blazor WASM (UI) ──JS interop──> módulos ES en wwwroot/js ──> ONNX Runtime Web (CDN)
                                      │
                                      ├─ WebGPU si el navegador lo soporta
                                      └─ WASM (CPU) como fallback automático
```

- **La inferencia es JavaScript, no .NET.** Blazor solo maneja UI y orquesta vía `IJSRuntime`.
- Cada modo tiene su módulo ES con su propio pipeline; los modelos se cargan una vez y se cachean en memoria.

### Estructura

```
PhotoRestore/
├── Pages/Restore.razor        # Única página ("/" y "/restore"): modos, factor, progreso, resultado
├── Layout/MainLayout.razor    # Layout mínimo sin barra lateral
├── wwwroot/
│   ├── js/
│   │   ├── upscaler.js        # Real-ESRGAN: tiles 128px + solape, reescalado x1/x2/x4
│   │   ├── face-restore.js    # SCRFD (detecta) → alineado ArcFace → GFPGAN → pegado suave
│   │   ├── colorize.js        # DDColor: RGB→Lab, modelo a 512², AB reescalado, Lab→RGB
│   │   ├── remove-bg.js       # RMBG-1.4: máscara de sujeto a 1024², min-max → alpha
│   │   └── ui.js              # Animaciones GSAP, dropzone drag&drop, hint del comparador
│   ├── models/                # Modelos ONNX (ver su README)
│   ├── css/app-custom.css     # Tema oscuro, glassmorphism, variables de paleta
│   ├── lottie/loading.json    # Animación de "procesando" (autoalojada)
│   └── index.html             # CDNs: onnxruntime-web, GSAP, lottie-player, img-comparison-slider
└── models-backup/             # Originales sin parchear (excluido de git)
```

## Decisiones técnicas clave

- **WebGPU con fallback WASM explícito**: si cualquier parte de la pipeline falla en WebGPU (no solo la creación de sesión, también la primera inferencia), se liberan las sesiones y se reintenta todo en WASM sin redescargar el modelo. La UI muestra qué motor se usó.
- **Carga de modelo manual**: el `.onnx` se descarga con `fetch` propio (`no-cache`, validación de tamaño, 1 reintento) y la sesión se crea desde bytes. Esto convirtió el críptico "Failed to fetch" en mensajes diagnósticos claros.
- **Modelos en fp32, no fp16**: el backend WASM/CPU de onnxruntime-web no tiene kernels fp16 para Conv; fp16 rompería el fallback.
- **SCRFD parcheado** (`ceil_mode` 1→0 en 3 AveragePool): el EP WebGPU no soporta `ceil()` en shape computation. Verificado bit a bit idéntico (diff = 0.0).
- **Scores SCRFD**: el grafo ya incluye los `Sigmoid` — no aplicar sigmoide de nuevo en JS.
- **Tiling** (solo Real-ESRGAN): tiles de 128px con solape de 10px para acotar memoria; los márgenes se descartan al componer (×4). Es por memoria, no por velocidad.
- **Alineado de caras**: transformación de semejanza (Umeyama) de los 5 landmarks de SCRFD a la plantilla ArcFace ×4; pegado con máscara elipse con feather (`destination-in`).

## Pendiente de verificación en navegador

Lo que nunca se pudo probar automatizado (sin navegador automatizable en el entorno de desarrollo):

- **DDColor y RMBG-1.4**: integrados y verificados visualmente en navegador por el usuario (funcionan).
- Preprocesado SCRFD confirmado solo por convención (BGR, [-1,1]) — validado en la práctica por el usuario.
- Responsive móvil y aspecto del handle personalizado del comparador.

## Notas para producción (despliegue público)

- **HTTPS obligatorio**: WebGPU requiere contexto seguro (localhost ya lo es). GitHub Pages / Cloudflare Pages sirven Blazor WASM standalone gratis.
- **Autoalojar los CDNs**: onnxruntime-web, GSAP, lottie-player e img-comparison-slider se cargan de jsdelivr con versión fijada; en producción conviene copiarlos a `wwwroot/lib/` (hay comentarios en `index.html`).
- **Modelos grandes** (GFPGAN 340 MB, DDColor 258 MB, RMBG 176 MB): GFPGAN y RMBG están en `.gitignore` con URL de descarga documentada en `wwwroot/models/README.md`; DDColor está commiteado. Servir con compresión Brotli y cache agresivo (Cache API / IndexedDB para no redescargar).
- **Licencias**: revisar antes de uso comercial — **RMBG-1.4 es solo para uso no comercial** (licencia bria-rmbg-1.4; alternativa Apache 2.0: U²-Net); GFPGAN arrastra componentes de terceros con posible restricción comercial; SCRFD y DDColor son Apache 2.0; la animación Lottie es LottieFiles Community License (conviene atribución).

## Roadmap discutido

- **Próximos modelos candidatos** (evaluados en sesión, por orden de valor/esfuerzo): **CodeFormer** (alternativa a GFPGAN para caras muy degradadas, con parámetro fidelidad/calidad), **Real-ESRGAN Anime** (variante para ilustraciones; reutiliza el pipeline de `upscaler.js`), **FastSAM/MobileSAM + LaMa** ("borrador mágico": segmentación por clic + inpainting; requiere UI de canvas/máscara), **Zero-DCE** (mejora de fotos con poca luz, <1 MB).
- **RMBG-1.4 (quitar fondo): implementado.** Limitación conocida: la selección del sujeto es automática; para elegir qué quitar haría falta la vía FastSAM + LaMa de arriba.
- **Restauración de video**: analizado y en pausa. Requiere backend con GPU (yt-dlp/ffmpeg + batching), ya no cabe en el navegador; frame-a-frame con Real-ESRGAN produce parpadeo temporal (la solución sería RealBasicVSR/BasicVSR++). Referencia: https://github.com/k4yt3x/video2x. Ojo: descargar de YouTube viola sus ToS para un servicio público; la variante segura es upload de video propio.
