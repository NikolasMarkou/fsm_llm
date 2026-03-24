// FSM-LLM Monitor — Keyboard Shortcuts & Initialization

'use strict';

document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
        case '1': showPage('dashboard'); break;
        case '2': showPage('control'); break;
        case '3': showPage('visualizer'); break;
        case '4': showPage('logs'); break;
        case '5': showPage('settings'); break;
        case 'Escape':
            closeLaunchModal();
            if (typeof closeDrawer === 'function') closeDrawer();
            break;
    }
});

// === BOOT ===

connectWS();
loadSettings();
refreshInstances();
setInterval(updateClock, 1000);
updateClock();
setInterval(function() {
    if (App.currentPage === 'control' || App.currentPage === 'dashboard') {
        refreshInstances();
        if (App.currentPage === 'control') refreshControlCenter();
    }
}, 5000);
