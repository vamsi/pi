/**
 * Collapsible thinking block component.
 */

import { createMarkdownElement } from './markdown-renderer.js';

/**
 * Render a thinking block.
 * @param {string} content - Thinking content
 * @param {boolean} isStreaming
 * @returns {HTMLElement}
 */
export function renderThinkingBlock(content, isStreaming = false) {
    const block = document.createElement('div');
    block.className = 'thinking-block';

    const header = document.createElement('div');
    header.className = 'thinking-header flex items-center gap-2 py-1 text-sm text-muted-foreground hover:text-foreground transition-colors';

    const chevron = document.createElement('span');
    chevron.className = 'transition-transform inline-block text-xs';
    chevron.textContent = '\u25B6'; // right-pointing triangle

    const label = document.createElement('span');
    if (isStreaming) {
        label.className = 'animate-shimmer bg-gradient-to-r from-muted-foreground via-foreground to-muted-foreground bg-[length:200%_100%] bg-clip-text text-transparent';
    }
    label.textContent = 'Thinking...';

    header.appendChild(chevron);
    header.appendChild(label);

    const body = document.createElement('div');
    body.className = 'hidden mt-2';
    body.appendChild(createMarkdownElement(content, true));

    let expanded = false;
    header.addEventListener('click', () => {
        expanded = !expanded;
        body.classList.toggle('hidden', !expanded);
        chevron.style.transform = expanded ? 'rotate(90deg)' : '';
    });

    block.appendChild(header);
    block.appendChild(body);
    return block;
}
