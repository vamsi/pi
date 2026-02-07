/**
 * Tool call display component (collapsible).
 */

/**
 * Render a tool call message.
 * @param {object} toolCall - ToolCall object {id, name, arguments}
 * @param {object} [options]
 * @param {object} [options.result] - ToolResultMessage
 * @param {boolean} [options.pending]
 * @param {boolean} [options.aborted]
 * @param {boolean} [options.isStreaming]
 * @returns {HTMLElement}
 */
export function renderToolMessage(toolCall, options = {}) {
    const { result, pending = false, aborted = false, isStreaming = false } = options;
    const card = document.createElement('div');
    card.className = 'tool-card';

    // Header
    const header = document.createElement('div');
    header.className = 'flex items-center justify-between gap-2 text-sm text-muted-foreground cursor-pointer';

    const left = document.createElement('div');
    left.className = 'flex items-center gap-2';

    // Status icon
    const status = document.createElement('span');
    if (pending || isStreaming) {
        status.className = 'inline-block w-3 h-3 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin';
    } else if (result?.isError || aborted) {
        status.className = 'text-destructive';
        status.textContent = '\u2717'; // X mark
    } else if (result) {
        status.className = 'text-green-500';
        status.textContent = '\u2713'; // checkmark
    } else {
        status.textContent = '\u2022'; // bullet
    }
    left.appendChild(status);

    const name = document.createElement('span');
    name.textContent = toolCall.name || 'Tool call';
    left.appendChild(name);

    const chevron = document.createElement('span');
    chevron.className = 'text-xs transition-transform';
    chevron.textContent = '\u25BC'; // down arrow

    header.appendChild(left);
    header.appendChild(chevron);

    // Body (hidden by default)
    const body = document.createElement('div');
    body.className = 'hidden mt-3 flex flex-col gap-2';

    // Arguments
    const args = toolCall.arguments || {};
    if (Object.keys(args).length > 0) {
        const argsLabel = document.createElement('div');
        argsLabel.className = 'text-xs font-medium text-muted-foreground';
        argsLabel.textContent = 'Arguments';
        body.appendChild(argsLabel);

        const argsCode = document.createElement('pre');
        argsCode.className = 'text-xs bg-muted p-2 rounded overflow-x-auto';
        argsCode.textContent = JSON.stringify(args, null, 2);
        body.appendChild(argsCode);
    }

    // Result
    if (result) {
        const resultLabel = document.createElement('div');
        resultLabel.className = 'text-xs font-medium text-muted-foreground';
        resultLabel.textContent = result.isError ? 'Error' : 'Result';
        body.appendChild(resultLabel);

        const resultText = (result.content || [])
            .filter(c => c.type === 'text')
            .map(c => c.text)
            .join('\n');

        if (resultText) {
            const resultCode = document.createElement('pre');
            resultCode.className = `text-xs p-2 rounded overflow-x-auto ${result.isError ? 'bg-destructive/10 text-destructive' : 'bg-muted'}`;
            resultCode.textContent = resultText;
            body.appendChild(resultCode);
        }
    }

    if (aborted) {
        const abortedEl = document.createElement('div');
        abortedEl.className = 'text-xs text-destructive italic';
        abortedEl.textContent = 'Aborted';
        body.appendChild(abortedEl);
    }

    // Toggle
    let expanded = false;
    header.addEventListener('click', () => {
        expanded = !expanded;
        body.classList.toggle('hidden', !expanded);
        chevron.style.transform = expanded ? 'rotate(180deg)' : '';
    });

    card.appendChild(header);
    card.appendChild(body);
    return card;
}
