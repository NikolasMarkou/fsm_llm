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

function showToast(msg, type) {
    var prev = document.querySelector('.toast');
    if (prev) prev.remove();
    var toast = document.createElement('div');
    toast.className = 'toast toast-' + (type || 'error');
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(function() { toast.classList.add('toast-visible'); }, 10);
    setTimeout(function() {
        toast.classList.remove('toast-visible');
        setTimeout(function() { toast.remove(); }, 300);
    }, 4000);
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

function copyContextData() {
    var data = App._lastContextData;
    if (!data) return;
    var json = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(json).then(function() {
        var btn = document.querySelector('#conv-context-kv').previousElementSibling.querySelector('.btn');
        if (btn) {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('btn-primary');
            setTimeout(function() { btn.textContent = orig; btn.classList.remove('btn-primary'); }, 1500);
        }
    }).catch(function() {
        // Fallback for non-HTTPS contexts
        var ta = document.createElement('textarea');
        ta.value = json;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    });
}

function formatNumber(n) {
    if (n == null || isNaN(n)) return '0';
    n = Number(n);
    if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
    if (n >= 1000) return n.toLocaleString('en-US');
    return String(n);
}

async function fetchJson(url, opts) {
    var resp = await fetch(url, opts);
    if (!resp.ok) {
        var body;
        try { body = await resp.json(); } catch (e) { body = null; }
        var detail = (body && body.detail) ? body.detail : resp.statusText;
        throw new Error(detail);
    }
    return resp.json();
}

function highlightText(text, query) {
    if (!query) return esc(text);
    var escaped = esc(text);
    var escapedQuery = esc(query);
    var re = new RegExp('(' + escapedQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    return escaped.replace(re, '<span class="search-highlight">$1</span>');
}
