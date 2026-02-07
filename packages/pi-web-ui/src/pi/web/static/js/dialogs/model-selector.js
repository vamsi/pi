/**
 * Model/provider picker dialog.
 */

/**
 * Show the model selector dialog.
 * @param {Array} providers - [{name, models: [{id, name, reasoning, input, cost, contextWindow, maxTokens}]}]
 * @param {object|null} currentModel
 * @param {Function} onSelect - (provider, modelId) => void
 */
export function showModelSelector(providers, currentModel, onSelect) {
    const dialog = document.createElement('dialog');
    dialog.className = 'bg-card text-card-foreground border border-border rounded-lg shadow-xl backdrop:bg-black/50 max-w-md w-full max-h-[90vh] flex flex-col';

    let searchQuery = '';
    let filterThinking = false;
    let filterVision = false;

    function getFilteredModels() {
        const all = [];
        for (const p of providers) {
            for (const m of p.models) {
                all.push({ provider: p.name, ...m });
            }
        }

        let filtered = all;

        if (searchQuery) {
            const tokens = searchQuery.toLowerCase().split(/\s+/).filter(Boolean);
            filtered = filtered.filter(m => {
                const text = `${m.provider} ${m.id} ${m.name || ''}`.toLowerCase();
                return tokens.every(t => text.includes(t));
            });
        }

        if (filterThinking) {
            filtered = filtered.filter(m => m.reasoning);
        }
        if (filterVision) {
            filtered = filtered.filter(m => m.input?.includes('image'));
        }

        // Sort: current model first
        filtered.sort((a, b) => {
            const aCurrent = currentModel && a.id === currentModel.id && a.provider === currentModel.provider;
            const bCurrent = currentModel && b.id === currentModel.id && b.provider === currentModel.provider;
            if (aCurrent && !bCurrent) return -1;
            if (!aCurrent && bCurrent) return 1;
            return a.provider.localeCompare(b.provider);
        });

        return filtered;
    }

    function formatTokens(n) {
        if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)}M`;
        if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
        return String(n);
    }

    function render() {
        const models = getFilteredModels();

        dialog.innerHTML = `
            <div class="p-4 pb-3 border-b border-border flex-shrink-0">
                <h2 class="text-lg font-semibold mb-3">Select Model</h2>
                <input id="model-search" type="text" class="w-full px-3 py-2 rounded-md border border-input bg-background text-foreground text-sm mb-2" placeholder="Search models..." value="${searchQuery}">
                <div class="flex gap-2">
                    <button id="filter-thinking" class="text-xs px-2 py-1 rounded-full border ${filterThinking ? 'bg-primary text-primary-foreground border-primary' : 'border-border text-muted-foreground hover:text-foreground'}">
                        Thinking
                    </button>
                    <button id="filter-vision" class="text-xs px-2 py-1 rounded-full border ${filterVision ? 'bg-primary text-primary-foreground border-primary' : 'border-border text-muted-foreground hover:text-foreground'}">
                        Vision
                    </button>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto">
                ${models.map(m => {
                    const isCurrent = currentModel && m.id === currentModel.id && m.provider === currentModel.provider;
                    const cost = m.cost;
                    const costStr = (cost?.input || cost?.output) ? `$${(cost.input || 0).toFixed(2)}/$${(cost.output || 0).toFixed(2)}` : '';
                    return `
                        <div class="model-item px-4 py-3 hover:bg-muted cursor-pointer border-b border-border" data-provider="${m.provider}" data-model-id="${m.id}">
                            <div class="flex items-center justify-between gap-2 mb-1">
                                <div class="flex items-center gap-2 min-w-0">
                                    <span class="text-sm font-medium text-foreground truncate">${m.id}</span>
                                    ${isCurrent ? '<span class="text-green-500 text-xs">&#10003;</span>' : ''}
                                </div>
                                <span class="text-xs px-1.5 py-0.5 rounded border border-border text-muted-foreground whitespace-nowrap">${m.provider}</span>
                            </div>
                            <div class="flex items-center justify-between text-xs text-muted-foreground">
                                <div class="flex items-center gap-2">
                                    <span class="${m.reasoning ? '' : 'opacity-30'}" title="Thinking">&#129504;</span>
                                    <span class="${m.input?.includes('image') ? '' : 'opacity-30'}" title="Vision">&#128065;</span>
                                    <span>${formatTokens(m.contextWindow || 0)}K/${formatTokens(m.maxTokens || 0)}K</span>
                                </div>
                                <span>${costStr}</span>
                            </div>
                        </div>
                    `;
                }).join('')}
                ${models.length === 0 ? '<div class="p-4 text-center text-muted-foreground text-sm">No models found</div>' : ''}
            </div>
        `;

        // Bind events
        dialog.querySelector('#model-search')?.addEventListener('input', (e) => {
            searchQuery = e.target.value;
            render();
            dialog.querySelector('#model-search')?.focus();
        });

        dialog.querySelector('#filter-thinking')?.addEventListener('click', () => {
            filterThinking = !filterThinking;
            render();
        });

        dialog.querySelector('#filter-vision')?.addEventListener('click', () => {
            filterVision = !filterVision;
            render();
        });

        dialog.querySelectorAll('.model-item').forEach(el => {
            el.addEventListener('click', () => {
                onSelect(el.dataset.provider, el.dataset.modelId);
                dialog.close();
                dialog.remove();
            });
        });
    }

    render();

    dialog.addEventListener('close', () => dialog.remove());
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) {
            dialog.close();
            dialog.remove();
        }
    });

    document.body.appendChild(dialog);
    dialog.showModal();
    dialog.querySelector('#model-search')?.focus();
}
