// FSM-LLM Monitor — Formatting Utilities

/** Format a timestamp to HH:MM:SS. */
export function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts).substring(11, 19);
    return d.toLocaleTimeString('en-US', { hour12: false });
}

/** Human-readable relative time (e.g., "5m ago"). */
export function relativeTime(dateStr) {
    if (!dateStr) return '';
    const diff = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
    if (diff < 5) return 'just now';
    if (diff < 60) return diff + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
}

/** Humanize large numbers (1234 -> "1,234", 12345 -> "12.3K"). */
export function formatNumber(n) {
    if (n == null || isNaN(n)) return '0';
    n = Number(n);
    if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
    if (n >= 1000) return n.toLocaleString('en-US');
    return String(n);
}
