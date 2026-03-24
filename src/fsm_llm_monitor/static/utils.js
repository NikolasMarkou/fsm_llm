// FSM-LLM Monitor — Utility Functions

'use strict';

function scheduleRefresh(key, fn, delayMs) {
    if (App.refreshTimers[key]) return;
    App.refreshTimers[key] = setTimeout(function() {
        App.refreshTimers[key] = null;
        fn();
    }, delayMs);
}

function esc(s) {
    if (s == null) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function formatTime(ts) {
    if (!ts) return '';
    var d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts).substring(11, 19);
    return d.toLocaleTimeString('en-US', { hour12: false });
}

function relativeTime(dateStr) {
    if (!dateStr) return '';
    var diff = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
    if (diff < 5) return 'just now';
    if (diff < 60) return diff + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
}

// Legacy alias
function _relativeTime(dateStr) { return relativeTime(dateStr); }

function updateClock() {
    var el = document.getElementById('clock');
    var el2 = document.getElementById('footer-clock');
    var t = new Date().toLocaleTimeString('en-US', { hour12: false });
    if (el) el.textContent = t;
    if (el2) el2.textContent = t;
}

function numVal(id, fallback) {
    var v = parseFloat(document.getElementById(id).value);
    return Number.isFinite(v) ? v : fallback;
}

function intVal(id, fallback) {
    var v = parseInt(document.getElementById(id).value, 10);
    return Number.isFinite(v) ? v : fallback;
}

function showError(elementId, msg) {
    var el = document.getElementById(elementId);
    if (el) el.innerHTML = '<span class="error-message">' + esc(msg) + '</span>';
}

function renderResultBanner(success) {
    var cls = success ? 'success' : 'failure';
    var text = success ? 'Agent completed successfully' : 'Agent failed';
    return '<div class="result-banner ' + cls + '">' + text + '</div>';
}

function showStatus(elementId, msg, color) {
    var el = document.getElementById(elementId);
    if (el) el.innerHTML = '<span class="status-msg status-' + color + '">' + esc(msg) + '</span>';
}

function statusBadge(status) {
    var cls = 'badge-' + status;
    return '<span class="badge ' + cls + '">' + esc(status.toUpperCase()) + '</span>';
}

function _renderLLMData(obj) {
    if (!obj || typeof obj !== 'object') return '<span class="text-dim">No data</span>';
    var html = '';
    var keys = Object.keys(obj);
    for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        var v = obj[k];
        if (v === null || v === undefined) continue;
        var display;
        if (typeof v === 'object') {
            display = '<pre>' + esc(JSON.stringify(v, null, 2)) + '</pre>';
        } else {
            display = esc(String(v));
        }
        html += '<div class="llm-kv"><span class="llm-key">' + esc(k) + ':</span> ' + display + '</div>';
    }
    return html || '<span class="text-dim">Empty</span>';
}

function highlightText(text, query) {
    if (!query) return esc(text);
    var escaped = esc(text);
    var escapedQuery = esc(query);
    var re = new RegExp('(' + escapedQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    return escaped.replace(re, '<span class="search-highlight">$1</span>');
}
