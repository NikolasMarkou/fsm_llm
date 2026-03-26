// FSM-LLM Monitor — Graph Layout & SVG Rendering
// BFS-based layered layout with adaptive SVG sizing.

import { esc } from './dom.js';

const W = 180, H = 60;

function layoutNodes(nodes, edges) {
    if (!nodes?.length) return;

    const adj = {};
    for (const n of nodes) adj[n.id] = [];
    for (const e of edges) {
        if (adj[e.from] && e.from !== e.to) adj[e.from].push(e.to);
    }

    let start = nodes[0].id;
    for (const n of nodes) { if (n.is_initial) { start = n.id; break; } }

    const layers = {};
    const visited = new Set([start]);
    const queue = [start];
    layers[start] = 0;

    while (queue.length) {
        const cur = queue.shift();
        for (const n of (adj[cur] || [])) {
            if (!visited.has(n)) {
                visited.add(n);
                layers[n] = (layers[cur] || 0) + 1;
                queue.push(n);
            }
        }
    }

    let maxLayer = 0;
    for (const k in layers) if (layers[k] > maxLayer) maxLayer = layers[k];
    for (const n of nodes) if (layers[n.id] === undefined) layers[n.id] = maxLayer + 1;

    const layerGroups = {};
    for (const n of nodes) {
        const l = layers[n.id];
        (layerGroups[l] ??= []).push(n);
    }

    const XPAD = 120, YPAD = 100, MAX_COLS = 5;
    const layerKeys = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
    const wrapRow = layerKeys.length > MAX_COLS ? MAX_COLS : layerKeys.length;

    for (let li = 0; li < layerKeys.length; li++) {
        const group = layerGroups[layerKeys[li]];
        const col = li % wrapRow;
        const row = Math.floor(li / wrapRow);
        const effectiveCol = (row % 2 === 0) ? col : (wrapRow - 1 - col);

        let rowHeight = 0;
        for (let ri = row * wrapRow; ri < Math.min((row + 1) * wrapRow, layerKeys.length); ri++) {
            const g = layerGroups[layerKeys[ri]];
            if (g.length > rowHeight) rowHeight = g.length;
        }

        const x = 140 + effectiveCol * (W + XPAD);
        const rowYBase = 60 + row * (rowHeight * (H + YPAD) + 60);
        for (let ni = 0; ni < group.length; ni++) {
            group[ni].x = x;
            group[ni].y = rowYBase + ni * (H + YPAD);
        }
    }
}

function rectEdgePoint(cx, cy, tx, ty, w, h) {
    const dx = tx - cx, dy = ty - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    const hw = w / 2, hh = h / 2;
    const sx = dx !== 0 ? hw / Math.abs(dx) : Infinity;
    const sy = dy !== 0 ? hh / Math.abs(dy) : Infinity;
    const s = Math.min(sx, sy);
    return { x: cx + dx * s, y: cy + dy * s };
}

export function renderGraph(svgId, data, opts = {}) {
    const svg = document.getElementById(svgId);
    if (!svg) return;

    const { nodes, edges } = data;
    const colorVar = opts.colorVar || 'var(--primary-dim)';
    const arrowColor = opts.arrowColor || colorVar;
    const rx = opts.rx || 4;
    const markerId = 'arrow-' + svgId;
    const nodeClass = opts.nodeClass || 'fsm';

    layoutNodes(nodes, edges);

    const nodeMap = {};
    for (const n of nodes) nodeMap[n.id] = n;

    // Calculate viewBox bounds
    let minX = Infinity, minY = Infinity, maxX = 0, maxY = 0;
    for (const n of nodes) {
        if (n.x - W / 2 < minX) minX = n.x - W / 2;
        if (n.y - H / 2 < minY) minY = n.y - H / 2;
        if (n.x + W / 2 > maxX) maxX = n.x + W / 2;
        if (n.y + H / 2 > maxY) maxY = n.y + H / 2;
    }
    minY -= 60; // padding for self-loop arcs

    const contentW = maxX - minX;
    const contentH = maxY - minY;
    const PAD = 60;
    const svgW = contentW + PAD * 2;
    const svgH = contentH + PAD * 2;
    const offsetX = PAD - minX;
    const offsetY = PAD - minY;
    for (const n of nodes) { n.x += offsetX; n.y += offsetY; }

    const container = svg.parentElement;
    const containerW = container?.clientWidth || svgW;
    const containerH = container?.clientHeight || svgH;
    const vbW = Math.max(svgW, containerW);
    const vbH = Math.max(svgH, containerH);
    const extraX = (vbW - contentW - PAD * 2) / 2;
    const extraY = (vbH - contentH - PAD * 2) / 2;
    if (extraX > 0 || extraY > 0) {
        for (const n of nodes) { n.x += extraX; n.y += extraY; }
    }

    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('viewBox', `0 0 ${vbW} ${vbH}`);
    svg.style.minWidth = svgW > containerW ? svgW + 'px' : '';
    svg.style.minHeight = svgH > containerH ? svgH + 'px' : '';

    let html = `<defs><marker id="${markerId}" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="${arrowColor}"/></marker></defs>`;

    // Count bidirectional edges
    const edgePairs = {};
    for (const e of edges) {
        const key = [e.from, e.to].sort().join('|');
        edgePairs[key] = (edgePairs[key] || 0) + 1;
    }

    // Draw edges
    const drawnPairs = {};
    for (const e of edges) {
        const from = nodeMap[e.from], to = nodeMap[e.to];
        if (!from || !to) continue;

        if (e.from === e.to) {
            const cx = from.x, cy = from.y - H / 2;
            html += `<path class="edge-line" d="M ${cx - 20} ${cy} C ${cx - 30} ${cy - 50}, ${cx + 30} ${cy - 50}, ${cx + 20} ${cy}" fill="none" marker-end="url(#${markerId})"/>`;
            if (e.label) html += `<text class="edge-label" x="${cx}" y="${cy - 38}">${esc(e.label)}</text>`;
            continue;
        }

        const pairKey = [e.from, e.to].sort().join('|');
        const isBidi = edgePairs[pairKey] > 1;
        let offset = 0;
        if (isBidi) {
            drawnPairs[pairKey] = (drawnPairs[pairKey] || 0);
            offset = drawnPairs[pairKey] === 0 ? 12 : -12;
            drawnPairs[pairKey]++;
        }

        const p1 = rectEdgePoint(from.x, from.y, to.x, to.y, W, H);
        const p2 = rectEdgePoint(to.x, to.y, from.x, from.y, W + 16, H + 16);

        if (offset !== 0) {
            const dx = to.x - from.x, dy = to.y - from.y;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const nx = -dy / len * offset, ny = dx / len * offset;
            const mx = (p1.x + p2.x) / 2 + nx, my = (p1.y + p2.y) / 2 + ny;
            html += `<path class="edge-line" d="M ${p1.x} ${p1.y} Q ${mx} ${my} ${p2.x} ${p2.y}" fill="none" marker-end="url(#${markerId})"/>`;
            if (e.label) html += `<text class="edge-label" x="${mx}" y="${my - 14}">${esc(e.label)}</text>`;
        } else {
            html += `<line class="edge-line" x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" marker-end="url(#${markerId})"/>`;
            if (e.label) html += `<text class="edge-label" x="${(p1.x + p2.x) / 2}" y="${(p1.y + p2.y) / 2 - 14}">${esc(e.label)}</text>`;
        }
    }

    // Draw nodes
    for (const n of nodes) {
        const cls = n.is_initial ? 'initial' : n.is_terminal ? 'terminal' : '';
        html += `<rect class="node-rect node-${nodeClass} ${cls}" x="${n.x - W / 2}" y="${n.y - H / 2}" width="${W}" height="${H}" rx="${rx}"/>`;
        html += `<text class="node-label" x="${n.x}" y="${n.y - 6}">${esc(n.label || n.id)}</text>`;
        const subtitle = n.step_type || (n.description ? n.description.substring(0, 24) : '');
        if (subtitle) html += `<text class="node-subtitle" x="${n.x}" y="${n.y + 12}">${esc(subtitle)}</text>`;
    }

    svg.innerHTML = html;
}
