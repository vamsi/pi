/**
 * Markdown artifact renderer.
 */

import { createMarkdownElement } from '../components/markdown-renderer.js';

/**
 * Render a Markdown artifact.
 * @param {object} artifact - {filename, content, title}
 * @returns {HTMLElement}
 */
export function renderMarkdownArtifact(artifact) {
    const wrapper = document.createElement('div');
    wrapper.className = 'p-6 max-w-3xl mx-auto';
    wrapper.appendChild(createMarkdownElement(artifact.content || ''));
    return wrapper;
}
