/**
 * Text/code artifact renderer with syntax highlighting.
 */

/**
 * Render a text/code artifact.
 * @param {object} artifact - {filename, content, title}
 * @returns {HTMLElement}
 */
export function renderTextArtifact(artifact) {
    const wrapper = document.createElement('div');
    wrapper.className = 'w-full h-full flex flex-col';

    // Header with filename and copy button
    const header = document.createElement('div');
    header.className = 'flex items-center justify-between px-4 py-2 border-b border-border bg-muted';

    const filename = document.createElement('span');
    filename.className = 'text-xs text-muted-foreground font-mono';
    filename.textContent = artifact.filename || 'file';
    header.appendChild(filename);

    const copyBtn = document.createElement('button');
    copyBtn.className = 'text-xs text-muted-foreground hover:text-foreground px-2 py-0.5 rounded hover:bg-secondary transition-colors';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(artifact.content || '').then(() => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        });
    });
    header.appendChild(copyBtn);

    // Code content
    const codeWrapper = document.createElement('div');
    codeWrapper.className = 'flex-1 overflow-auto';

    const pre = document.createElement('pre');
    pre.className = 'p-4 text-sm';

    const code = document.createElement('code');

    // Try to detect language from file extension
    const ext = (artifact.filename || '').split('.').pop()?.toLowerCase() || '';
    const langMap = {
        js: 'javascript', ts: 'typescript', py: 'python', rb: 'ruby',
        rs: 'rust', go: 'go', java: 'java', cpp: 'cpp', c: 'c',
        css: 'css', json: 'json', xml: 'xml', yaml: 'yaml', yml: 'yaml',
        sh: 'bash', sql: 'sql', jsx: 'javascript', tsx: 'typescript',
    };
    const lang = langMap[ext] || '';

    code.textContent = artifact.content || '';
    if (lang && typeof hljs !== 'undefined') {
        try {
            code.innerHTML = hljs.highlight(artifact.content || '', { language: lang }).value;
        } catch (e) { /* fallback to plain text */ }
    }

    pre.appendChild(code);
    codeWrapper.appendChild(pre);

    wrapper.appendChild(header);
    wrapper.appendChild(codeWrapper);
    return wrapper;
}
