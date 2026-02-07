/**
 * API key entry modal.
 */

/**
 * Show API key prompt dialog.
 * @param {string} provider
 * @param {Function} onSave - (provider, key) => void
 */
export function showApiKeyPrompt(provider, onSave) {
    const dialog = document.getElementById('api-key-dialog');
    const providerEl = document.getElementById('api-key-provider');
    const input = document.getElementById('api-key-input');

    providerEl.textContent = provider;
    input.value = '';
    dialog.showModal();
    input.focus();

    // The save/cancel handlers are set up in app.js
}
