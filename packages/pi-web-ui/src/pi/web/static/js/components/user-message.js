/**
 * User message bubble component.
 */

import { createMarkdownElement } from './markdown-renderer.js';

/**
 * Render a user message.
 * @param {object} message - User message object
 * @returns {HTMLElement}
 */
export function renderUserMessage(message) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-start mx-4';

    const bubble = document.createElement('div');
    bubble.className = 'user-message-bubble';

    // Extract text content
    let text = '';
    if (typeof message.content === 'string') {
        text = message.content;
    } else if (Array.isArray(message.content)) {
        text = message.content
            .filter(c => c.type === 'text')
            .map(c => c.text)
            .join('');
    }

    bubble.appendChild(createMarkdownElement(text));

    // Attachments
    if (message.attachments && message.attachments.length > 0) {
        const attachments = document.createElement('div');
        attachments.className = 'mt-2 flex flex-wrap gap-2';
        for (const att of message.attachments) {
            const tile = document.createElement('div');
            tile.className = 'text-xs px-2 py-1 rounded bg-muted text-muted-foreground';
            tile.textContent = att.fileName || att.file_name || 'file';
            attachments.appendChild(tile);
        }
        bubble.appendChild(attachments);
    }

    wrapper.appendChild(bubble);
    return wrapper;
}
