/**
 * Formatting utilities for tokens and costs.
 */

/** Format a token count to a human-readable string */
export function formatTokenCount(tokens) {
    if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
    if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}K`;
    return String(tokens);
}

/** Format a cost in dollars */
export function formatCost(cost) {
    if (cost === 0) return '$0';
    if (cost < 0.01) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
}

/** Format usage stats */
export function formatUsage(usage) {
    if (!usage) return '';
    const parts = [];
    const totalTokens = (usage.input || 0) + (usage.output || 0);
    if (totalTokens > 0) {
        parts.push(`${formatTokenCount(totalTokens)} tokens`);
    }
    const cost = usage.cost?.total || 0;
    if (cost > 0) {
        parts.push(formatCost(cost));
    }
    return parts.join(' · ');
}

/** Format model cost (per million tokens) */
export function formatModelCost(cost) {
    if (!cost) return '';
    const input = cost.input || 0;
    const output = cost.output || 0;
    if (input === 0 && output === 0) return 'Free';
    return `$${input.toFixed(2)}/$${output.toFixed(2)} /M`;
}

/** Get aggregate usage from messages */
export function getAggregateUsage(messages) {
    const totals = {
        input: 0, output: 0, cacheRead: 0, cacheWrite: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    };

    for (const msg of messages) {
        if (msg.role === 'assistant' && msg.usage) {
            totals.input += msg.usage.input || 0;
            totals.output += msg.usage.output || 0;
            totals.cacheRead += msg.usage.cacheRead || 0;
            totals.cacheWrite += msg.usage.cacheWrite || 0;
            if (msg.usage.cost) {
                totals.cost.total += msg.usage.cost.total || 0;
            }
        }
    }
    return totals;
}
