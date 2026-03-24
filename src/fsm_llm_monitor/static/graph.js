// FSM-LLM Monitor — Graph Layout & Rendering

'use strict';

function layoutNodes(nodes, edges) {
    if (!nodes || nodes.length === 0) return;
    var nodeMap = {};
    for (var i = 0; i < nodes.length; i++) nodeMap[nodes[i].id] = nodes[i];

    var adj = {};
    for (var i = 0; i < nodes.length; i++) adj[nodes[i].id] = [];
    for (var i = 0; i < edges.length; i++) {
        if (adj[edges[i].from] && edges[i].from !== edges[i].to) {
            adj[edges[i].from].push(edges[i].to);
        }
    }

    var start = nodes[0].id;
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].is_initial) { start = nodes[i].id; break; }
    }

    var layers = {};
    var visited = {};
    var queue = [start];
    visited[start] = true;
    layers[start] = 0;
    while (queue.length > 0) {
        var cur = queue.shift();
        var neighbors = adj[cur] || [];
        for (var i = 0; i < neighbors.length; i++) {
            var n = neighbors[i];
            if (!visited[n]) {
                visited[n] = true;
                layers[n] = (layers[cur] || 0) + 1;
                queue.push(n);
            }
        }
    }
    var maxLayer = 0;
    for (var k in layers) { if (layers[k] > maxLayer) maxLayer = layers[k]; }
    for (var i = 0; i < nodes.length; i++) {
        if (layers[nodes[i].id] === undefined) {
            layers[nodes[i].id] = maxLayer + 1;
        }
    }

    var layerGroups = {};
    for (var i = 0; i < nodes.length; i++) {
        var l = layers[nodes[i].id];
        if (!layerGroups[l]) layerGroups[l] = [];
        layerGroups[l].push(nodes[i]);
    }

    var W = 180, H = 60, XPAD = 120, YPAD = 100;
    var layerKeys = Object.keys(layerGroups).map(Number).sort(function(a, b) { return a - b; });

    var MAX_COLS = 5;
    var totalLayers = layerKeys.length;
    var wrapRow = totalLayers > MAX_COLS ? MAX_COLS : totalLayers;

    for (var li = 0; li < layerKeys.length; li++) {
        var group = layerGroups[layerKeys[li]];
        var col = li % wrapRow;
        var row = Math.floor(li / wrapRow);
        var effectiveCol = (row % 2 === 0) ? col : (wrapRow - 1 - col);
        var rowHeight = 0;
        for (var ri = row * wrapRow; ri < Math.min((row + 1) * wrapRow, layerKeys.length); ri++) {
            var g = layerGroups[layerKeys[ri]];
            if (g.length > rowHeight) rowHeight = g.length;
        }
        var x = 140 + effectiveCol * (W + XPAD);
        var rowYBase = 60 + row * (rowHeight * (H + YPAD) + 60);
        for (var ni = 0; ni < group.length; ni++) {
            group[ni].x = x;
            group[ni].y = rowYBase + ni * (H + YPAD);
        }
    }
}

function rectEdgePoint(cx, cy, tx, ty, W, H) {
    var dx = tx - cx, dy = ty - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    var hw = W / 2, hh = H / 2;
    var sx = dx !== 0 ? hw / Math.abs(dx) : Infinity;
    var sy = dy !== 0 ? hh / Math.abs(dy) : Infinity;
    var s = Math.min(sx, sy);
    return { x: cx + dx * s, y: cy + dy * s };
}

function renderGraph(svgId, data, opts) {
    opts = opts || {};
    var svg = document.getElementById(svgId);
    if (!svg) return;
    var nodes = data.nodes;
    var edges = data.edges;
    var colorVar = opts.colorVar || 'var(--primary-dim)';
    var arrowColor = opts.arrowColor || colorVar;
    var rx = opts.rx || 4;
    var markerId = 'arrow-' + svgId;
    var W = 180, H = 60;

    layoutNodes(nodes, edges);

    var nodeMap = {};
    for (var i = 0; i < nodes.length; i++) nodeMap[nodes[i].id] = nodes[i];

    var minX = Infinity, minY = Infinity, maxX = 0, maxY = 0;
    for (var i = 0; i < nodes.length; i++) {
        var nx = nodes[i].x, ny = nodes[i].y;
        if (nx - W / 2 < minX) minX = nx - W / 2;
        if (ny - H / 2 < minY) minY = ny - H / 2;
        if (nx + W / 2 > maxX) maxX = nx + W / 2;
        if (ny + H / 2 > maxY) maxY = ny + H / 2;
    }
    // Add padding for self-loop arcs and edge labels above top nodes
    minY = minY - 60;
    var contentW = maxX - minX;
    var contentH = maxY - minY;
    var PAD = 60;
    var svgW = contentW + PAD * 2;
    var svgH = contentH + PAD * 2;
    // Shift all node positions so content is centered in the viewBox
    var offsetX = PAD - minX;
    var offsetY = PAD - minY;
    for (var i = 0; i < nodes.length; i++) {
        nodes[i].x += offsetX;
        nodes[i].y += offsetY;
    }
    // Use container width for wider centering on small graphs
    var container = svg.parentElement;
    var containerW = container ? container.clientWidth : svgW;
    if (svgW < containerW) svgW = containerW;
    var containerH = container ? container.clientHeight : svgH;
    if (svgH < containerH) svgH = containerH;
    // Re-center after expanding to container size
    var extraX = (svgW - contentW - PAD * 2) / 2;
    var extraY = (svgH - contentH - PAD * 2) / 2;
    if (extraX > 0 || extraY > 0) {
        for (var i = 0; i < nodes.length; i++) {
            nodes[i].x += extraX;
            nodes[i].y += extraY;
        }
    }
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.setAttribute('viewBox', '0 0 ' + svgW + ' ' + svgH);

    var html = '<defs><marker id="' + markerId + '" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="' + arrowColor + '"/></marker></defs>';

    var edgePairs = {};
    for (var i = 0; i < edges.length; i++) {
        var key = [edges[i].from, edges[i].to].sort().join('|');
        if (!edgePairs[key]) edgePairs[key] = 0;
        edgePairs[key]++;
    }

    var drawnPairs = {};
    for (var i = 0; i < edges.length; i++) {
        var e = edges[i];
        var from = nodeMap[e.from];
        var to = nodeMap[e.to];
        if (!from || !to) continue;
        if (e.from === e.to) {
            var cx = from.x, cy = from.y - H / 2;
            html += '<path class="edge-line" d="M ' + (cx - 20) + ' ' + cy + ' C ' + (cx - 30) + ' ' + (cy - 50) + ', ' + (cx + 30) + ' ' + (cy - 50) + ', ' + (cx + 20) + ' ' + cy + '" fill="none" marker-end="url(#' + markerId + ')"/>';
            if (e.label) html += '<text class="edge-label" x="' + cx + '" y="' + (cy - 38) + '">' + esc(e.label) + '</text>';
            continue;
        }

        var pairKey = [e.from, e.to].sort().join('|');
        var isBidi = edgePairs[pairKey] > 1;
        var offset = 0;
        if (isBidi) {
            if (!drawnPairs[pairKey]) drawnPairs[pairKey] = 0;
            offset = drawnPairs[pairKey] === 0 ? 12 : -12;
            drawnPairs[pairKey]++;
        }

        var p1 = rectEdgePoint(from.x, from.y, to.x, to.y, W, H);
        var p2 = rectEdgePoint(to.x, to.y, from.x, from.y, W + 16, H + 16);

        if (offset !== 0) {
            var dx = to.x - from.x, dy = to.y - from.y;
            var len = Math.sqrt(dx * dx + dy * dy) || 1;
            var nx = -dy / len * offset, ny = dx / len * offset;
            var mx = (p1.x + p2.x) / 2 + nx, my = (p1.y + p2.y) / 2 + ny;
            html += '<path class="edge-line" d="M ' + p1.x + ' ' + p1.y + ' Q ' + mx + ' ' + my + ' ' + p2.x + ' ' + p2.y + '" fill="none" marker-end="url(#' + markerId + ')"/>';
            if (e.label) html += '<text class="edge-label" x="' + mx + '" y="' + (my - 14) + '">' + esc(e.label) + '</text>';
        } else {
            html += '<line class="edge-line" x1="' + p1.x + '" y1="' + p1.y + '" x2="' + p2.x + '" y2="' + p2.y + '" marker-end="url(#' + markerId + ')"/>';
            if (e.label) {
                html += '<text class="edge-label" x="' + ((p1.x + p2.x) / 2) + '" y="' + ((p1.y + p2.y) / 2 - 14) + '">' + esc(e.label) + '</text>';
            }
        }
    }

    var nodeClass = opts.nodeClass || 'fsm';
    for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var cls = n.is_initial ? 'initial' : n.is_terminal ? 'terminal' : '';
        html += '<rect class="node-rect node-' + nodeClass + ' ' + cls + '" x="' + (n.x - W / 2) + '" y="' + (n.y - H / 2) + '" width="' + W + '" height="' + H + '" rx="' + rx + '"/>';
        html += '<text class="node-label" x="' + n.x + '" y="' + (n.y - 6) + '">' + esc(n.label || n.id) + '</text>';
        var subtitle = n.step_type || (n.description ? n.description.substring(0, 24) : '');
        if (subtitle) {
            html += '<text class="node-subtitle" x="' + n.x + '" y="' + (n.y + 12) + '">' + esc(subtitle) + '</text>';
        }
    }

    svg.innerHTML = html;
}
