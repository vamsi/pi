/**
 * Settings dialog — API keys management.
 */

/**
 * Show the settings dialog.
 * @param {Array} providers - provider names
 * @param {Function} onSaveKey - (provider, key) => void
 * @param {Function} onDeleteKey - (provider) => void
 */
export function showSettingsDialog(providers, onSaveKey, onDeleteKey) {
    const dialog = document.createElement('dialog');
    dialog.className = 'bg-card text-card-foreground border border-border rounded-lg shadow-xl backdrop:bg-black/50 max-w-lg w-full max-h-[90vh] flex flex-col';

    dialog.innerHTML = `
        <div class="p-4 border-b border-border flex-shrink-0">
            <h2 class="text-lg font-semibold">Settings</h2>
            <p class="text-sm text-muted-foreground mt-1">Configure API keys for LLM providers</p>
        </div>
        <div class="flex-1 overflow-y-auto p-4 space-y-4">
            ${providers.map(name => `
                <div class="flex flex-col gap-2">
                    <label class="text-sm font-medium text-foreground">${name}</label>
                    <div class="flex gap-2">
                        <input type="password" class="api-key-input flex-1 px-3 py-2 rounded-md border border-input bg-background text-foreground text-sm" placeholder="API key..." data-provider="${name}">
                        <button class="save-key-btn px-3 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:opacity-90 transition-opacity" data-provider="${name}">Save</button>
                    </div>
                </div>
            `).join('')}
            ${providers.length === 0 ? '<div class="text-center py-8 text-muted-foreground text-sm">No providers registered</div>' : ''}
        </div>
        <div class="p-4 border-t border-border flex justify-end">
            <button class="close-btn px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary transition-colors">Close</button>
        </div>
    `;

    // Bind events
    dialog.querySelectorAll('.save-key-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const provider = btn.dataset.provider;
            const input = dialog.querySelector(`.api-key-input[data-provider="${provider}"]`);
            const key = input?.value?.trim();
            if (key && provider) {
                onSaveKey(provider, key);
                input.value = '';
                btn.textContent = 'Saved!';
                setTimeout(() => { btn.textContent = 'Save'; }, 1500);
            }
        });
    });

    dialog.querySelector('.close-btn')?.addEventListener('click', () => {
        dialog.close();
        dialog.remove();
    });

    dialog.addEventListener('close', () => dialog.remove());
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) {
            dialog.close();
            dialog.remove();
        }
    });

    document.body.appendChild(dialog);
    dialog.showModal();
}
