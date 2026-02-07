/**
 * Client-side state store.
 */

export class State {
    constructor() {
        this.sessionId = '';
        this.model = null;
        this.thinkingLevel = 'off';
        /** @type {Array} */
        this.messages = [];
        this.isStreaming = false;
        /** Current streaming message being built */
        this.streamMessage = null;
        /** @type {Array} */
        this.providers = [];
        /** @type {Array} */
        this.sessions = [];
        /** @type {Set<Function>} */
        this._listeners = new Set();
    }

    /** Subscribe to state changes. Returns unsubscribe fn. */
    subscribe(fn) {
        this._listeners.add(fn);
        return () => this._listeners.delete(fn);
    }

    /** Notify all subscribers */
    notify() {
        for (const fn of this._listeners) {
            try { fn(this); } catch (e) { console.error('State listener error:', e); }
        }
    }

    /** Apply a full state update from server */
    applyState(data) {
        this.sessionId = data.sessionId || '';
        this.model = data.model || null;
        this.thinkingLevel = data.thinkingLevel || 'off';
        this.messages = data.messages || [];
        this.isStreaming = data.isStreaming || false;
        this.streamMessage = null;
        this.notify();
    }

    /** Apply models list from server */
    applyModels(data) {
        this.providers = data.providers || [];
        this.notify();
    }

    /** Apply sessions list from server */
    applySessions(data) {
        this.sessions = data.sessions || [];
        this.notify();
    }

    /** Start streaming */
    setStreaming(streaming) {
        this.isStreaming = streaming;
        if (!streaming) {
            this.streamMessage = null;
        }
        this.notify();
    }

    /** Update the current stream message */
    setStreamMessage(message) {
        this.streamMessage = message;
        this.notify();
    }

    /** Add a completed message to the list */
    addMessage(message) {
        this.messages = [...this.messages, message];
        this.notify();
    }
}
