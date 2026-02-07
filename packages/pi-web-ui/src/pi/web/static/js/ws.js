/**
 * WebSocket client with auto-reconnect.
 */

export class WebSocketClient {
    constructor() {
        /** @type {WebSocket|null} */
        this._ws = null;
        /** @type {Map<string, Set<Function>>} */
        this._listeners = new Map();
        this._reconnectDelay = 1000;
        this._maxReconnectDelay = 30000;
        this._currentDelay = this._reconnectDelay;
        this._shouldReconnect = true;
        this._connected = false;
    }

    /** Connect to the WebSocket server */
    connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${location.host}/ws`;

        this._ws = new WebSocket(url);

        this._ws.onopen = () => {
            this._connected = true;
            this._currentDelay = this._reconnectDelay;
            this._emit('open', null);
        };

        this._ws.onclose = () => {
            this._connected = false;
            this._emit('close', null);
            if (this._shouldReconnect) {
                setTimeout(() => this.connect(), this._currentDelay);
                this._currentDelay = Math.min(this._currentDelay * 2, this._maxReconnectDelay);
            }
        };

        this._ws.onerror = (err) => {
            this._emit('error', err);
        };

        this._ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._emit('message', data);
                // Also emit by message type
                if (data.type) {
                    this._emit(data.type, data);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };
    }

    /** Send a JSON message */
    send(data) {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            this._ws.send(JSON.stringify(data));
        }
    }

    /** Subscribe to an event type */
    on(event, callback) {
        if (!this._listeners.has(event)) {
            this._listeners.set(event, new Set());
        }
        this._listeners.get(event).add(callback);
        return () => this._listeners.get(event)?.delete(callback);
    }

    /** Disconnect */
    disconnect() {
        this._shouldReconnect = false;
        if (this._ws) {
            this._ws.close();
        }
    }

    get connected() {
        return this._connected;
    }

    _emit(event, data) {
        const listeners = this._listeners.get(event);
        if (listeners) {
            for (const fn of listeners) {
                try { fn(data); } catch (e) { console.error('Event handler error:', e); }
            }
        }
    }
}
