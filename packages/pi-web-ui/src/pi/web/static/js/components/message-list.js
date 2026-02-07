/**
 * Message list renderer — renders all completed messages.
 */

import { renderUserMessage } from './user-message.js';
import { renderAssistantMessage } from './assistant-message.js';

/**
 * Render the full message list into a container.
 * @param {HTMLElement} container
 * @param {Array} messages
 * @param {object} [options]
 * @param {Set} [options.pendingToolCalls]
 * @param {boolean} [options.isStreaming]
 */
export function renderMessageList(container, messages, options = {}) {
    const { pendingToolCalls = new Set(), isStreaming = false } = options;

    // Build a map of tool results by call ID
    const toolResultsById = new Map();
    for (const msg of messages) {
        if (msg.role === 'tool_result' || msg.role === 'toolResult') {
            const id = msg.toolCallId || msg.tool_call_id;
            if (id) toolResultsById.set(id, msg);
        }
    }

    container.innerHTML = '';

    for (const msg of messages) {
        if (msg.role === 'user' || msg.role === 'user-with-attachments') {
            container.appendChild(renderUserMessage(msg));
        } else if (msg.role === 'assistant') {
            container.appendChild(renderAssistantMessage(msg, {
                isStreaming: false,
                toolResultsById,
                pendingToolCalls,
            }));
        }
        // tool_result messages are rendered inline with assistant messages
    }
}

/**
 * Build tool results map from messages.
 */
export function buildToolResultsMap(messages) {
    const map = new Map();
    for (const msg of messages) {
        if (msg.role === 'tool_result' || msg.role === 'toolResult') {
            const id = msg.toolCallId || msg.tool_call_id;
            if (id) map.set(id, msg);
        }
    }
    return map;
}
