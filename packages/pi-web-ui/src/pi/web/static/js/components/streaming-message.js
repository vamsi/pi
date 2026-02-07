/**
 * Streaming message display — updates live during agent streaming.
 */

import { renderAssistantMessage } from './assistant-message.js';

/**
 * Update the streaming message container.
 * @param {HTMLElement} container - The streaming container element
 * @param {object|null} message - Current streaming message
 * @param {boolean} isStreaming
 * @param {Map} toolResultsById
 * @param {Set} pendingToolCalls
 */
export function updateStreamingMessage(container, message, isStreaming, toolResultsById = new Map(), pendingToolCalls = new Set()) {
    const msgEl = container.querySelector('#streaming-message');
    const indicator = container.querySelector('.animate-pulse')?.parentElement;

    if (!message || !isStreaming) {
        container.classList.add('hidden');
        if (msgEl) msgEl.innerHTML = '';
        return;
    }

    container.classList.remove('hidden');

    if (message.role === 'assistant') {
        if (msgEl) {
            msgEl.innerHTML = '';
            msgEl.appendChild(renderAssistantMessage(message, { isStreaming: true, toolResultsById, pendingToolCalls }));
        }
    }

    // Show/hide cursor
    if (indicator) {
        indicator.style.display = isStreaming ? '' : 'none';
    }
}
