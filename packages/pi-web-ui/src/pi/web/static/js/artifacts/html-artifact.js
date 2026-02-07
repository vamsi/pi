/**
 * HTML artifact renderer — sandboxed iframe.
 */

/**
 * Render an HTML artifact in a sandboxed iframe.
 * @param {object} artifact - {filename, content, title}
 * @returns {HTMLElement}
 */
export function renderHtmlArtifact(artifact) {
    const wrapper = document.createElement('div');
    wrapper.className = 'w-full h-full';

    const iframe = document.createElement('iframe');
    iframe.sandbox = 'allow-scripts allow-modals';
    iframe.style.cssText = 'width: 100%; height: 100%; border: none;';
    iframe.srcdoc = artifact.content || '';

    wrapper.appendChild(iframe);
    return wrapper;
}
