/**
 * File attachment preview tile.
 */

/**
 * Create an attachment tile element.
 * @param {object} attachment - {id, fileName, mimeType, size}
 * @param {object} [options]
 * @param {boolean} [options.showDelete]
 * @param {Function} [options.onDelete]
 * @returns {HTMLElement}
 */
export function createAttachmentTile(attachment, options = {}) {
    const { showDelete = false, onDelete } = options;

    const tile = document.createElement('div');
    tile.className = 'relative group inline-block';

    const isImage = attachment.mimeType?.startsWith('image/');
    const fileName = attachment.fileName || 'file';

    if (isImage && attachment.previewUrl) {
        tile.innerHTML = `
            <img src="${attachment.previewUrl}" class="w-16 h-16 object-cover rounded-lg border border-input" alt="${fileName}" title="${fileName}">
        `;
    } else {
        // Document icon
        const ext = fileName.split('.').pop()?.toUpperCase() || 'FILE';
        tile.innerHTML = `
            <div class="w-16 h-16 rounded-lg border border-input bg-muted text-muted-foreground flex flex-col items-center justify-center p-2" title="${fileName}">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                </svg>
                <div class="text-[10px] text-center truncate w-full mt-1">${ext}</div>
            </div>
        `;
    }

    if (showDelete && onDelete) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'absolute -top-1 -right-1 w-5 h-5 bg-background hover:bg-muted text-muted-foreground hover:text-foreground rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity border border-input shadow-sm text-xs';
        deleteBtn.textContent = '\u2715';
        deleteBtn.title = 'Remove';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            onDelete(attachment.id);
        });
        tile.appendChild(deleteBtn);
    }

    return tile;
}
