/**
 * Assistant message component — renders text, thinking, and tool calls.
 */

import { createMarkdownElement } from './markdown-renderer.js';
import { renderToolMessage } from './tool-message.js';
import { renderThinkingBlock } from './thinking-block.js';
import { formatUsage } from '../utils/format.js';

/**
 * Render an assistant message.
 * @param {object} message - AssistantMessage object
 * @param {object} [options]
 * @param {boolean} [options.isStreaming]
 * @param {Map} [options.toolResultsById]
 * @param {Set} [options.pendingToolCalls]
 * @returns {HTMLElement}
 */
export function renderAssistantMessage(message, options = {}) {
    const { isStreaming = false, toolResultsById = new Map(), pendingToolCalls = new Set() } = options;
    const container = document.createElement('div');

    const content = message.content || [];
    const parts = document.createElement('div');
    parts.className = 'px-4 flex flex-col gap-3';

    for (const chunk of content) {
        if (chunk.type === 'text' && chunk.text?.trim()) {
            parts.appendChild(createMarkdownElement(chunk.text));
        } else if (chunk.type === 'thinking' && chunk.thinking?.trim()) {
            parts.appendChild(renderThinkingBlock(chunk.thinking, isStreaming));
        } else if (chunk.type === 'toolCall' || chunk.type === 'tool_call') {
            const toolCallId = chunk.id;
            const pending = pendingToolCalls.has(toolCallId);
            const result = toolResultsById.get(toolCallId);
            const aborted = message.stopReason === 'aborted' && !result;
            parts.appendChild(renderToolMessage(chunk, { result, pending, aborted, isStreaming }));
        }
    }

    if (parts.children.length > 0) {
        container.appendChild(parts);
    }

    // Usage stats
    if (message.usage && !isStreaming) {
        const usageText = formatUsage(message.usage);
        if (usageText) {
            const usage = document.createElement('div');
            usage.className = 'px-4 mt-2 text-xs text-muted-foreground';
            usage.textContent = usageText;
            container.appendChild(usage);
        }
    }

    // Error
    if (message.stopReason === 'error' && message.errorMessage) {
        const error = document.createElement('div');
        error.className = 'mx-4 mt-3 p-3 bg-destructive/10 text-destructive rounded-lg text-sm overflow-hidden';
        error.innerHTML = `<strong>Error:</strong> ${escapeHtml(message.errorMessage)}`;
        container.appendChild(error);
    }

    // Aborted
    if (message.stopReason === 'aborted') {
        const aborted = document.createElement('span');
        aborted.className = 'px-4 text-sm text-destructive italic';
        aborted.textContent = 'Request aborted';
        container.appendChild(aborted);
    }

    return container;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
