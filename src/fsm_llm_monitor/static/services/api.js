// FSM-LLM Monitor — HTTP Client
// Thin fetch wrapper with JSON error handling.

export async function fetchJson(url, opts) {
    const resp = await fetch(url, opts);
    if (!resp.ok) {
        let body;
        try { body = await resp.json(); } catch { body = null; }
        const detail = body?.detail ?? resp.statusText;
        throw new Error(detail);
    }
    return resp.json();
}

export function postJson(url, data) {
    return fetchJson(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
}
