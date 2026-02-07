/**
 * Main application controller.
 * Connects WebSocket, state, and UI components.
 */

import { WebSocketClient } from './ws.js';
import { State } from './state.js';
import { renderMessageList, buildToolResultsMap } from './components/message-list.js';
import { initMessageEditor } from './components/message-editor.js';
import { updateStreamingMessage } from './components/streaming-message.js';
import { formatUsage, getAggregateUsage } from './utils/format.js';
import { showModelSelector } from './dialogs/model-selector.js';
import { showSessionList } from './dialogs/session-list.js';
import { initTheme, toggleTheme } from './utils/theme.js';
import { renderArtifactsPanel } from './artifacts/artifacts-panel.js';
import { showSettingsDialog } from './dialogs/settings.js';

// --- Globals ---
const ws = new WebSocketClient();
const state = new State();
let editor = null;
let autoScroll = true;

// --- DOM refs ---
const messagesContainer = document.getElementById('messages-container');
const messageListEl = document.getElementById('message-list');
const streamingContainer = document.getElementById('streaming-container');
const editorContainer = document.getElementById('message-editor');
const statsEl = document.getElementById('stats');
const modelNameEl = document.getElementById('model-name');
const thinkingLevelEl = document.getElementById('thinking-level');
const btnNewSession = document.getElementById('btn-new-session');
const btnSessions = document.getElementById('btn-sessions');
const btnModel = document.getElementById('btn-model');
const btnThinking = document.getElementById('btn-thinking');
const btnTheme = document.getElementById('btn-theme');
const btnSettings = document.getElementById('btn-settings');

// API key dialog
const apiKeyDialog = document.getElementById('api-key-dialog');
const apiKeyProviderEl = document.getElementById('api-key-provider');
const apiKeyInput = document.getElementById('api-key-input');
const apiKeySave = document.getElementById('api-key-save');
const apiKeyCancel = document.getElementById('api-key-cancel');

// --- Init editor ---
editor = initMessageEditor(editorContainer, {
    onSend: (text, attachments = []) => {
        const attachmentIds = attachments.map(a => a.id);
        ws.send({ type: 'prompt', text, attachments: attachmentIds });
        autoScroll = true;
    },
    onAbort: () => {
        ws.send({ type: 'abort' });
    },
    getIsStreaming: () => state.isStreaming,
});

// --- Auto-scroll logic ---
messageListEl.addEventListener('scroll', () => {
    const { scrollTop, scrollHeight, clientHeight } = messageListEl;
    const distFromBottom = scrollHeight - scrollTop - clientHeight;
    if (distFromBottom > 50) {
        autoScroll = false;
    } else if (distFromBottom < 10) {
        autoScroll = true;
    }
});

function scrollToBottom() {
    if (autoScroll) {
        messageListEl.scrollTop = messageListEl.scrollHeight;
    }
}

// --- Render functions ---
function renderMessages() {
    renderMessageList(messagesContainer, state.messages, {
        pendingToolCalls: new Set(),
        isStreaming: state.isStreaming,
    });
    scrollToBottom();
}

function renderStreamingMessage() {
    const toolResultsById = buildToolResultsMap(state.messages);
    updateStreamingMessage(
        streamingContainer,
        state.streamMessage,
        state.isStreaming,
        toolResultsById,
        new Set(),
    );
    scrollToBottom();
}

function renderStats() {
    const usage = getAggregateUsage(state.messages);
    const text = formatUsage(usage);
    statsEl.textContent = text;
}

function renderHeader() {
    if (state.model) {
        modelNameEl.textContent = state.model.id || 'Unknown model';
        // Show thinking button if model supports reasoning
        if (state.model.reasoning) {
            btnThinking.classList.remove('hidden');
            thinkingLevelEl.textContent = state.thinkingLevel || 'off';
        } else {
            btnThinking.classList.add('hidden');
        }
    } else {
        modelNameEl.textContent = 'No model';
        btnThinking.classList.add('hidden');
    }
}

// --- WebSocket event handlers ---
ws.on('state', (data) => {
    state.applyState(data);
    renderMessages();
    renderHeader();
    renderStats();
    editor.updateState(state.isStreaming);
});

ws.on('models', (data) => {
    state.applyModels(data);
});

ws.on('sessions', (data) => {
    state.applySessions(data);
});

ws.on('agent_start', () => {
    state.setStreaming(true);
    editor.updateState(true);
});

ws.on('agent_end', () => {
    state.setStreaming(false);
    state.streamMessage = null;
    streamingContainer.classList.add('hidden');
    editor.updateState(false);
    renderMessages();
    renderStats();
});

ws.on('message_start', (data) => {
    state.setStreamMessage(data.message);
    renderStreamingMessage();
});

ws.on('message_update', (data) => {
    state.streamMessage = data.message;
    renderStreamingMessage();
});

ws.on('message_end', (data) => {
    state.streamMessage = null;
    state.messages = [...state.messages, data.message];
    streamingContainer.classList.add('hidden');
    renderMessages();
    renderStats();
});

ws.on('tool_start', (data) => {
    // Tool calls are rendered as part of the streaming assistant message
});

ws.on('tool_end', (data) => {
    // Tool results will appear in the next message_update/message_end
});

ws.on('error', (data) => {
    console.error('Server error:', data.message);
    // Show error as a transient notification
    showNotification(data.message, 'error');
});

ws.on('api_key_required', (data) => {
    showApiKeyDialog(data.provider);
});

ws.on('api_key_saved', () => {
    apiKeyDialog.close();
});

ws.on('artifacts', (data) => {
    const artifactsPanel = document.getElementById('artifacts-panel');
    const messagesPanel = document.getElementById('messages-panel');
    const artifacts = data.artifacts || [];

    if (artifacts.length > 0) {
        messagesPanel.style.width = '50%';
        renderArtifactsPanel(artifactsPanel, artifacts, () => {
            artifactsPanel.classList.add('hidden');
            messagesPanel.style.width = '100%';
        });
    } else {
        artifactsPanel.classList.add('hidden');
        messagesPanel.style.width = '100%';
    }
});

// --- API Key Dialog ---
function showApiKeyDialog(provider) {
    apiKeyProviderEl.textContent = provider;
    apiKeyInput.value = '';
    apiKeyDialog.showModal();
    apiKeyInput.focus();
}

apiKeySave.addEventListener('click', () => {
    const key = apiKeyInput.value.trim();
    const provider = apiKeyProviderEl.textContent;
    if (key && provider) {
        ws.send({ type: 'set_api_key', provider, key });
    }
});

apiKeyCancel.addEventListener('click', () => {
    apiKeyDialog.close();
});

apiKeyInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        apiKeySave.click();
    }
});

// --- Model selector ---
btnModel.addEventListener('click', () => {
    if (state.providers.length === 0) return;
    showModelSelector(state.providers, state.model, (provider, modelId) => {
        ws.send({ type: 'set_model', provider, modelId });
    });
});

// --- Thinking level selector ---
btnThinking.addEventListener('click', () => {
    const levels = ['off', 'minimal', 'low', 'medium', 'high'];
    const current = state.thinkingLevel || 'off';
    const idx = levels.indexOf(current);
    const next = levels[(idx + 1) % levels.length];
    ws.send({ type: 'set_thinking_level', level: next });
    state.thinkingLevel = next;
    thinkingLevelEl.textContent = next;
});

// --- Session management ---
btnNewSession.addEventListener('click', () => {
    ws.send({ type: 'new_session' });
    autoScroll = true;
});

btnSessions.addEventListener('click', () => {
    if (state.sessions.length === 0) {
        showNotification('No saved sessions', 'info');
        return;
    }
    showSessionList(
        state.sessions,
        (sessionId) => {
            ws.send({ type: 'load_session', sessionId });
            autoScroll = true;
        },
        (sessionId) => {
            ws.send({ type: 'delete_session', sessionId });
        },
    );
});

// --- Settings ---
btnSettings.addEventListener('click', () => {
    const providerNames = state.providers.map(p => p.name);
    showSettingsDialog(
        providerNames,
        (provider, key) => {
            ws.send({ type: 'set_api_key', provider, key });
        },
        (provider) => {
            ws.send({ type: 'delete_api_key', provider });
        },
    );
});

// --- Theme toggle ---
initTheme();
btnTheme.addEventListener('click', () => {
    toggleTheme();
});

// --- Notification helper ---
function showNotification(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `fixed top-4 right-4 z-50 px-4 py-2 rounded-lg shadow-lg text-sm max-w-sm ${
        type === 'error' ? 'bg-destructive text-destructive-foreground' : 'bg-secondary text-secondary-foreground'
    }`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

// --- Start ---
ws.connect();
editor.focus();
