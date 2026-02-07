/**
 * Markdown rendering using Marked.js + highlight.js.
 */

let markedConfigured = false;

function ensureConfigured() {
    if (markedConfigured) return;
    markedConfigured = true;

    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (e) { /* fallback */ }
            }
            try {
                return hljs.highlightAuto(code).value;
            } catch (e) { /* fallback */ }
            return code;
        },
        breaks: false,
        gfm: true,
    });
}

/** Render markdown string to HTML */
export function renderMarkdown(content) {
    if (!content) return '';
    ensureConfigured();
    return marked.parse(content);
}

/**
 * Create a markdown content element.
 * @param {string} content - Markdown string
 * @param {boolean} isThinking - Whether this is thinking content (dimmed)
 * @returns {HTMLElement}
 */
export function createMarkdownElement(content, isThinking = false) {
    const div = document.createElement('div');
    div.className = `markdown-content text-sm ${isThinking ? 'opacity-60' : ''}`;
    div.innerHTML = renderMarkdown(content);

    // Add copy buttons to code blocks
    div.querySelectorAll('pre').forEach(pre => {
        pre.style.position = 'relative';
        const btn = document.createElement('button');
        btn.className = 'code-copy-btn';
        btn.textContent = 'Copy';
        btn.addEventListener('click', () => {
            const code = pre.querySelector('code');
            navigator.clipboard.writeText(code?.textContent || '').then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            });
        });
        pre.appendChild(btn);
    });

    return div;
}
