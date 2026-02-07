/**
 * Artifacts panel — tab bar + artifact viewer.
 */

import { renderHtmlArtifact } from './html-artifact.js';
import { renderSvgArtifact } from './svg-artifact.js';
import { renderMarkdownArtifact } from './markdown-artifact.js';
import { renderTextArtifact } from './text-artifact.js';
import { renderImageArtifact } from './image-artifact.js';

/**
 * Render the artifacts panel.
 * @param {HTMLElement} container
 * @param {Array} artifacts - [{filename, content, title, version}]
 * @param {Function} onClose
 */
export function renderArtifactsPanel(container, artifacts, onClose) {
    if (!artifacts || artifacts.length === 0) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');
    container.className = 'w-1/2 border-l border-border flex flex-col h-full bg-background';

    let activeIndex = artifacts.length - 1;

    function render() {
        const artifact = artifacts[activeIndex];
        if (!artifact) return;

        container.innerHTML = `
            <div class="flex-shrink-0 border-b border-border">
                <div class="flex items-center justify-between px-3 py-2">
                    <div class="flex items-center gap-1 overflow-x-auto">
                        ${artifacts.map((a, i) => `
                            <button class="artifact-tab px-3 py-1 text-xs rounded whitespace-nowrap transition-colors ${
                                i === activeIndex
                                    ? 'bg-secondary text-foreground font-medium'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
                            }" data-index="${i}">
                                ${a.title || a.filename}
                            </button>
                        `).join('')}
                    </div>
                    <button class="close-btn ml-2 p-1 text-muted-foreground hover:text-foreground rounded hover:bg-secondary transition-colors" title="Close">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
            </div>
            <div id="artifact-content" class="flex-1 overflow-auto"></div>
        `;

        // Tab clicks
        container.querySelectorAll('.artifact-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                activeIndex = parseInt(tab.dataset.index, 10);
                render();
            });
        });

        // Close
        container.querySelector('.close-btn')?.addEventListener('click', onClose);

        // Render content
        const contentEl = container.querySelector('#artifact-content');
        renderArtifactContent(contentEl, artifact);
    }

    render();
}

function renderArtifactContent(container, artifact) {
    const filename = artifact.filename || '';
    const ext = filename.split('.').pop()?.toLowerCase() || '';

    container.innerHTML = '';

    switch (ext) {
        case 'html':
            container.appendChild(renderHtmlArtifact(artifact));
            break;
        case 'svg':
            container.appendChild(renderSvgArtifact(artifact));
            break;
        case 'md':
        case 'markdown':
            container.appendChild(renderMarkdownArtifact(artifact));
            break;
        case 'png':
        case 'jpg':
        case 'jpeg':
        case 'gif':
        case 'webp':
            container.appendChild(renderImageArtifact(artifact));
            break;
        default:
            container.appendChild(renderTextArtifact(artifact));
            break;
    }
}
