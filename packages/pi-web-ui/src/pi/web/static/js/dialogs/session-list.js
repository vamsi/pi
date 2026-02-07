/**
 * Session list dialog.
 */

/**
 * Show the session list dialog.
 * @param {Array} sessions - [{id, title, last_modified, message_count, preview}]
 * @param {Function} onSelect - (sessionId) => void
 * @param {Function} onDelete - (sessionId) => void
 */
export function showSessionList(sessions, onSelect, onDelete) {
    const dialog = document.createElement('dialog');
    dialog.className = 'bg-card text-card-foreground border border-border rounded-lg shadow-xl backdrop:bg-black/50 max-w-lg w-full max-h-[90vh] flex flex-col';

    function formatDate(isoString) {
        if (!isoString) return '';
        const date = new Date(isoString);
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        if (days === 0) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days} days ago`;
        return date.toLocaleDateString();
    }

    function render() {
        dialog.innerHTML = `
            <div class="p-4 border-b border-border flex-shrink-0">
                <h2 class="text-lg font-semibold">Sessions</h2>
                <p class="text-sm text-muted-foreground mt-1">Load a previous conversation</p>
            </div>
            <div class="flex-1 overflow-y-auto p-2 space-y-2">
                ${sessions.length === 0
                    ? '<div class="text-center py-8 text-muted-foreground text-sm">No sessions yet</div>'
                    : sessions.map(s => `
                        <div class="session-item group flex items-start gap-3 p-3 rounded-lg border border-border hover:bg-secondary/50 cursor-pointer transition-colors" data-id="${s.id}">
                            <div class="flex-1 min-w-0">
                                <div class="font-medium text-sm text-foreground truncate">${s.title || 'Untitled'}</div>
                                <div class="text-xs text-muted-foreground mt-1">${formatDate(s.last_modified)}</div>
                                <div class="text-xs text-muted-foreground mt-1">${s.message_count || 0} messages</div>
                                ${s.preview ? `<div class="text-xs text-muted-foreground mt-1 truncate">${s.preview}</div>` : ''}
                            </div>
                            <button class="delete-btn opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/10 text-destructive transition-opacity" data-id="${s.id}" title="Delete">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 6h18M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
                                </svg>
                            </button>
                        </div>
                    `).join('')
                }
            </div>
        `;

        // Bind events
        dialog.querySelectorAll('.session-item').forEach(el => {
            el.addEventListener('click', (e) => {
                if (e.target.closest('.delete-btn')) return;
                onSelect(el.dataset.id);
                dialog.close();
                dialog.remove();
            });
        });

        dialog.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('Delete this session?')) {
                    const id = btn.dataset.id;
                    onDelete(id);
                    // Remove from list
                    const idx = sessions.findIndex(s => s.id === id);
                    if (idx >= 0) sessions.splice(idx, 1);
                    render();
                }
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
}
