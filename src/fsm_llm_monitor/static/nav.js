// FSM-LLM Monitor — Navigation & Tabs

'use strict';

function showPage(page) {
    document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
    document.querySelectorAll('.sidebar-items button[data-page]').forEach(function(b) { b.classList.remove('active'); });
    var pageEl = document.getElementById('page-' + page);
    if (pageEl) pageEl.classList.add('active');
    var btn = document.querySelector('.sidebar-items button[data-page="' + page + '"]');
    if (btn) btn.classList.add('active');
    App.currentPage = page;

    // Close drawer when navigating away from control
    if (page !== 'control' && typeof closeDrawer === 'function') {
        closeDrawer();
    }

    var refreshMap = {
        'logs': refreshLogs,
        'settings': loadSettings,
        'control': refreshControlCenter
    };
    if (refreshMap[page]) refreshMap[page]();

    if (page === 'visualizer') {
        var activeTab = document.querySelector('.tab-content.active');
        if (activeTab && activeTab.id === 'tab-agents') {
            var sel = document.getElementById('viz-agent-type');
            if (sel && sel.value) visualizeGraph('agent', sel.value);
        } else if (activeTab && activeTab.id === 'tab-workflows') {
            var sel2 = document.getElementById('viz-wf-type');
            if (sel2 && sel2.value) visualizeGraph('workflow', sel2.value);
        }
    }
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

function switchTab(tabId, btn) {
    var tabBar = btn ? btn.parentElement : null;
    var scope = tabBar ? tabBar.parentElement : document;
    scope.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
    if (tabBar) tabBar.querySelectorAll('.tab').forEach(function(b) { b.classList.remove('active'); });
    var tabEl = document.getElementById(tabId);
    if (tabEl) tabEl.classList.add('active');
    if (btn) btn.classList.add('active');
}
