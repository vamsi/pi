/**
 * Image artifact renderer.
 */

/**
 * Render an image artifact.
 * @param {object} artifact - {filename, content, title}
 * @returns {HTMLElement}
 */
export function renderImageArtifact(artifact) {
    const wrapper = document.createElement('div');
    wrapper.className = 'w-full h-full flex items-center justify-center p-4';

    // If content is base64 data
    if (artifact.content?.startsWith('data:')) {
        const img = document.createElement('img');
        img.src = artifact.content;
        img.className = 'max-w-full max-h-full object-contain';
        img.alt = artifact.title || artifact.filename || 'image';
        wrapper.appendChild(img);
    } else {
        // Content is a URL or unsupported
        const placeholder = document.createElement('div');
        placeholder.className = 'text-muted-foreground text-sm';
        placeholder.textContent = `Image: ${artifact.filename || 'unknown'}`;
        wrapper.appendChild(placeholder);
    }

    return wrapper;
}
