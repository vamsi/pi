/**
 * SVG artifact renderer.
 */

/**
 * Render an SVG artifact.
 * @param {object} artifact - {filename, content, title}
 * @returns {HTMLElement}
 */
export function renderSvgArtifact(artifact) {
    const wrapper = document.createElement('div');
    wrapper.className = 'w-full h-full flex items-center justify-center p-4 overflow-auto';

    const container = document.createElement('div');
    container.className = 'max-w-full';
    container.innerHTML = artifact.content || '';

    // Ensure SVG scales properly
    const svg = container.querySelector('svg');
    if (svg) {
        svg.style.maxWidth = '100%';
        svg.style.height = 'auto';
    }

    wrapper.appendChild(container);
    return wrapper;
}
