// ui.js — Animaciones (GSAP) y dropzone de arrastrar y soltar para PhotoRestore.
// Todo lo que aquí se toca son elementos estables que Blazor no vuelve a
// renderizar (hero, tarjetas estáticas, dropzone); las secciones condicionales
// usan animación CSS (.fade-in) para no pelear con el renderizado de Blazor.

let animationsDone = false;

// Animación de entrada del hero y de las tarjetas marcadas con .anim-enter.
// Se ejecuta una sola vez por carga de página.
export function initAnimations() {
    if (animationsDone || typeof gsap === 'undefined') return;
    animationsDone = true;

    gsap.from('.hero > *', {
        opacity: 0, y: 24, duration: 0.7, stagger: 0.12, ease: 'power3.out'
    });
    gsap.from('.anim-enter', {
        opacity: 0, y: 30, duration: 0.6, stagger: 0.15, delay: 0.35, ease: 'power3.out'
    });
}

// Convierte el contenedor en zona de arrastrar y soltar. El <input type=file>
// de Blazor cubre toda la zona (CSS), así que el clic es nativo; aquí solo se
// resalta al arrastrar y se reenvían los archivos soltados al input mediante
// DataTransfer, disparando 'change' para que Blazor procese el archivo.
export function initDropzone(dropzone) {
    if (!dropzone || dropzone.dataset.dzInit) return;
    dropzone.dataset.dzInit = '1';

    const input = dropzone.querySelector('input[type=file]');
    if (!input) return;

    ['dragenter', 'dragover'].forEach(ev =>
        dropzone.addEventListener(ev, e => {
            e.preventDefault(); // necesario para permitir el drop
            dropzone.classList.add('drag-over');
        }));

    ['dragleave', 'drop'].forEach(ev =>
        dropzone.addEventListener(ev, e => {
            e.preventDefault(); // evita que el navegador navegue al archivo
            dropzone.classList.remove('drag-over');
        }));

    dropzone.addEventListener('drop', e => {
        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;
        try {
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            input.files = dt.files;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        } catch (err) {
            console.warn('No se pudo reenviar el archivo soltado al input.', err);
        }
    });
}
