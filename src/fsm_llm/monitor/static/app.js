// FSM-LLM Monitor — Main Entry Point (ES Module)
// Boot sequence, event delegation, keyboard shortcuts, navigation.

import { state } from './services/state.js';
import { connectWS, registerHandlers } from './services/ws.js';
import {
    getApiKey, checkAuthRequired, requestApiKey, onApiKeyChange,
    submitApiKeyModal, cancelApiKeyModal, isApiKeyModalOpen,
} from './services/auth.js';
import { $, openDialog, closeDialog } from './utils/dom.js';

// Page modules
import * as dashboard from './pages/dashboard.js';
import * as conversations from './pages/conversations.js';
import * as launch from './pages/launch.js';
import * as control from './pages/control.js';
import * as visualizer from './pages/visualizer.js';
import * as logs from './pages/logs.js';
import * as settings from './pages/settings.js';
import * as builder from './pages/builder.js';

// === NAVIGATION ===

const VALID_PAGES = ['dashboard', 'control', 'visualizer', 'logs', 'builder', 'settings'];

// Data (re)load hooks run when a page is shown or a new API key is saved.
const PAGE_REFRESH = {
    dashboard: () => { dashboard.loadDashboardConfig(); dashboard.refreshActivityTable(); },
    logs: () => logs.onShowLogs(),
    settings: () => settings.loadSettings(),
    control: () => control.refreshControlCenter(),
};

function showPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-items button[data-page]').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.mobile-nav-btn[data-page]').forEach(b => b.classList.remove('active'));

    const pageEl = $(`page-${page}`);
    if (pageEl) pageEl.classList.add('active');

    document.querySelector(`.sidebar-items button[data-page="${page}"]`)?.classList.add('active');
    document.querySelector(`.mobile-nav-btn[data-page="${page}"]`)?.classList.add('active');

    state.currentPage = page;

    if (location.hash !== `#${page}`) history.replaceState(null, '', `#${page}`);
    if (page !== 'control') control.closeDrawer();

    PAGE_REFRESH[page]?.();

    if (page === 'visualizer') {
        const activeTab = document.querySelector('.tab-content.active');
        if (activeTab?.id === 'tab-agents') {
            const sel = $('viz-agent-type');
            if (sel?.value) visualizer.visualizeGraph('agent', sel.value);
        } else if (activeTab?.id === 'tab-workflows') {
            const sel2 = $('viz-wf-type');
            if (sel2?.value) visualizer.visualizeGraph('workflow', sel2.value);
        }
    }
}

function toggleSidebar() {
    const collapsed = $('sidebar')?.classList.toggle('collapsed');
    $('sidebar-toggle')?.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
}

function toggleShortcuts(show) {
    const overlay = $('shortcuts-overlay');
    if (!overlay) return;
    const visible = overlay.style.display !== 'none';
    if (show ?? !visible) openDialog(overlay, 'flex');
    else closeDialog(overlay);
}

function switchTab(tabId, btn) {
    const tabBar = btn?.parentElement;
    const scope = tabBar?.parentElement || document;
    scope.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tabBar?.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    $(tabId)?.classList.add('active');
    btn?.classList.add('active');
}

function navigateFromHash() {
    const hash = location.hash.replace('#', '');
    if (hash && VALID_PAGES.includes(hash) && hash !== state.currentPage) showPage(hash);
}

// === CLOCK ===

function updateClock() {
    const t = new Date().toLocaleTimeString('en-US', { hour12: false });
    const el = $('clock');
    const el2 = $('footer-clock');
    if (el) el.textContent = t;
    if (el2) el2.textContent = t;
}

// === WIRE UP DEPENDENCIES ===

dashboard.setNavigateToInstance((id, type) => control.navigateToInstance(id, type));
conversations.setDeps({
    showPage,
    refreshActivityTable: dashboard.refreshActivityTable,
    refreshDetailPanel: control.refreshDetailPanel,
    openDrawer: control.openDrawer,
    refreshInstances: dashboard.refreshInstances,
});
launch.setDeps({
    showPage,
    refreshInstances: dashboard.refreshInstances,
    showConversationInDrawer: conversations.showConversationInDrawer,
});
control.setDeps({ showPage });
builder.setDeps({ showPage, refreshInstances: dashboard.refreshInstances });

// === REGISTER WS HANDLERS ===

registerHandlers({
    updateMetrics: dashboard.updateMetrics,
    updateEvents: dashboard.updateEvents,
    renderInstanceGrid: dashboard.renderInstanceGrid,
    renderUnifiedTable: control.renderUnifiedTable,
    updateRunningAgents: control.updateRunningAgents,
    appendLogs: logs.appendLogs,
    refreshActivityTable: dashboard.refreshActivityTable,
    showConversationDetail: conversations.showConversationDetail,
    refreshDetailPanel: control.refreshDetailPanel,
    dashboardConfigChanged: () => dashboard.loadDashboardConfig(),
});

// === EVENT DELEGATION ===
// All data-action clicks are dispatched here. No inline onclick handlers.

const ACTIONS = {
    // Navigation
    'show-page':        (_el, e) => showPage(e.target.closest('[data-page]')?.dataset.page),
    'toggle-sidebar':   () => toggleSidebar(),
    'switch-tab':       (_el, e) => { const btn = e.target.closest('[data-tab]'); switchTab(btn?.dataset.tab, btn); },

    // Dashboard
    'show-launch-modal': () => launch.showLaunchModal(),
    'navigate-instance': (el) => control.navigateToInstance(el.dataset.instanceId, el.dataset.instanceType),
    'inst-page-prev':    () => dashboard.instPagePrev(),
    'inst-page-next':    () => dashboard.instPageNext(),
    'activity-page-prev':  () => dashboard.activityPagePrev(),
    'activity-page-next':  () => dashboard.activityPageNext(),
    'toggle-activity-ended': () => dashboard.toggleActivityEnded(),
    'clear-dashboard-config': () => dashboard.clearDashboardConfig(),
    'activity-row-click':  (el) => {
        const itemType = el.dataset.itemType;
        const instanceId = el.dataset.instanceId;
        const itemId = el.dataset.itemId;
        if (itemType === 'fsm_conversation') {
            control.navigateToInstance(instanceId, 'fsm');
            setTimeout(() => conversations.showConversationInDrawer(instanceId, itemId), 200);
        } else if (itemType === 'agent_task') {
            control.navigateToInstance(instanceId, 'agent');
        } else if (itemType === 'workflow_instance') {
            control.navigateToInstance(instanceId, 'workflow');
        }
    },

    // Control Center
    'filter-instances':  (el) => control.filterControlInstances(el.dataset.filter, el),
    'open-drawer':       (el) => {
        const row = el.closest('[data-instance-id]');
        if (row) control.openDrawer(row.dataset.instanceId, row.dataset.instanceType);
    },
    'close-drawer':      () => control.closeDrawer(),
    'drawer-back':       () => conversations.drawerBack(),
    'start-conv':        (el) => control.startConversationOn(el.dataset.instanceId),
    'cancel-agent':      (el) => control.cancelAgent(el.dataset.instanceId),
    'advance-workflow':  (el) => control.advanceWorkflow(el.dataset.instanceId, el.dataset.wfInstanceId),
    'cancel-workflow':   (el) => control.cancelWorkflow(el.dataset.instanceId, el.dataset.wfInstanceId),
    'send-workflow-event': (el) => control.sendWorkflowEvent(el.dataset.instanceId),
    'end-conversation':  (el) => conversations.endConversation(el.dataset.instanceId, el.dataset.convId),
    'destroy-instance':  (el) => control.destroyInstance(el.dataset.instanceId),
    'go-to-conv':        (el) => conversations.showConversationInDrawer(el.dataset.instanceId, el.dataset.convId),
    'ctrl-page-prev':    () => control.ctrlPagePrev(),
    'ctrl-page-next':    () => control.ctrlPageNext(),
    'toggle-trace-step': (el) => control.toggleTraceStep(el),
    'expand-all-trace':  () => control.toggleAllTraceSteps(true),
    'collapse-all-trace':() => control.toggleAllTraceSteps(false),

    // Conversations
    'send-chat':         () => conversations.sendChatMessage(),
    'copy-context':      () => conversations.copyContextData(),

    // Visualizer
    'visualize-fsm':     () => visualizer.visualizeFSM(),
    'switch-viz-detail': (el) => visualizer.switchVizDetail(el.dataset.detail, el),

    // Logs
    'toggle-log-pill':   (el) => logs.toggleLogPill(el),
    'refresh-logs':      () => logs.refreshLogs(),
    'toggle-log-pause':  () => logs.toggleLogPause(),
    'clear-logs':        () => logs.clearLogs(),
    'log-jump-latest':   () => logs.logJumpToLatest(),

    // Builder
    'start-builder':        () => builder.startBuilderSession(),
    'reset-builder':        () => builder.resetBuilder(),
    'send-builder-message':  () => builder.sendBuilderMessage(),
    'builder-jump-latest':   () => builder.builderJumpToLatest(),
    'copy-builder-result':   () => builder.copyBuilderResult(),
    'download-builder-result': () => builder.downloadBuilderResult(),
    'launch-builder-result': () => builder.launchBuilderResult(),

    // Settings
    'save-settings':     () => settings.saveSettings(),
    'reset-settings':    () => settings.resetSettings(),
    'save-api-key':      () => settings.saveApiKeySetting(),
    'clear-api-key':     () => settings.clearApiKeySetting(),

    // Modals. Backdrop handlers compare the click target with the element that
    // carries the data-action (the overlay itself): the delegated listener sits
    // on document, so e.currentTarget is document and never matches.
    'close-launch-modal': () => launch.closeLaunchModal(),
    'close-shortcuts':    () => toggleShortcuts(false),
    'close-modal-backdrop': (el, e) => { if (e.target === el) launch.closeLaunchModal(); },
    'close-shortcuts-backdrop': (el, e) => { if (e.target === el) toggleShortcuts(false); },
    'apikey-save':        () => submitApiKeyModal(),
    'apikey-cancel':      () => cancelApiKeyModal(),
    'close-apikey-backdrop': (el, e) => { if (e.target === el) cancelApiKeyModal(); },

    // Launch modal
    'filter-presets':    (el) => launch.filterPresets(el.dataset.cat),
    'select-preset':     (el) => launch.selectPreset(el.closest('.preset-card')),
    'launch-fsm':        (el) => launch.doLaunchFSM(el),
    'launch-workflow':   (el) => launch.doLaunchWorkflow(el),
    'launch-agent':      (el) => launch.doLaunchAgent(el),
    'add-tool-from-template': () => launch.addToolFromTemplate(),
    'add-stub-tool':     () => launch.addStubTool(),
    'remove-stub-tool':  (el) => el.closest('.stub-tool-row')?.remove(),
};

document.addEventListener('click', (e) => {
    const actionEl = e.target.closest('[data-action]');
    if (!actionEl) return;

    const action = actionEl.dataset.action;

    // Stop propagation for nested actions (buttons inside clickable rows)
    const parentAction = actionEl.parentElement?.closest('[data-action]');
    if (parentAction && parentAction !== actionEl) e.stopPropagation();

    const handler = ACTIONS[action];
    if (handler) handler(actionEl, e);
});

// === CHANGE/INPUT EVENT DELEGATION ===

document.addEventListener('input', (e) => {
    const el = e.target;
    if (el.id === 'inst-search') dashboard.onInstSearchInput();
    else if (el.id === 'activity-search') dashboard.onActivitySearchInput();
    else if (el.id === 'ctrl-search') control.onCtrlSearchInput();
    else if (el.id === 'log-filter') logs.onLogSearchInput();
});

document.addEventListener('change', (e) => {
    const el = e.target;
    if (el.id === 'viz-preset-select') visualizer.useFSMPreset(el.value);
    else if (el.id === 'viz-agent-type') visualizer.visualizeGraph('agent', el.value);
    else if (el.id === 'viz-wf-type') visualizer.visualizeGraph('workflow', el.value);
    else if (el.id === 'launch-agent-type') launch.onAgentTypeChange();
    else if (el.id === 'launch-fsm-source') launch.toggleLaunchFSMSource();
});

// === KEYBOARD SHORTCUTS ===

const _NATIVE_ACTIVATABLE = ['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA', 'SUMMARY'];

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // The API key modal sits on top: close only it.
        if (isApiKeyModalOpen()) { cancelApiKeyModal(); return; }
        toggleShortcuts(false);
        launch.closeLaunchModal();
        control.closeDrawer();
        return;
    }
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
        if (e.key === 'Enter' && !e.shiftKey) {
            if (e.target.id === 'conv-message-input') { e.preventDefault(); conversations.sendChatMessage(); }
            else if (e.target.id === 'builder-message-input') { e.preventDefault(); builder.sendBuilderMessage(); }
            else if (e.target.id === 'log-filter') { logs.refreshLogs(); }
            else if (e.target.id === 'apikey-input') { e.preventDefault(); submitApiKeyModal(); }
            else if (e.target.id === 'set-api-key') { e.preventDefault(); settings.saveApiKeySetting(); }
        }
        return;
    }
    // Keyboard activation for non-native clickable elements (cards and rows
    // rendered with role="button" tabindex="0"): route through the same
    // click delegation as a mouse click.
    if ((e.key === 'Enter' || e.key === ' ') && e.target.dataset?.action
        && !_NATIVE_ACTIVATABLE.includes(tag)) {
        e.preventDefault();
        e.target.click();
        return;
    }
    if (isApiKeyModalOpen()) return;
    const pageKeys = { '1': 'dashboard', '2': 'control', '3': 'visualizer', '4': 'logs', '5': 'builder', '6': 'settings' };
    if (pageKeys[e.key]) { showPage(pageKeys[e.key]); return; }
    if (e.key === '?') toggleShortcuts();
});

// === SCROLL HANDLERS ===

$('log-stream')?.addEventListener('scroll', () => logs.onLogScroll());
$('builder-chat')?.addEventListener('scroll', () => builder.onBuilderScroll());

// === HASH ROUTING ===

window.addEventListener('hashchange', navigateFromHash);

// === BOOT SEQUENCE ===

// Ask the server whether a key is required before anything else, so the key
// modal opens once up front instead of on the first 401. Requests issued
// meanwhile share the same prompt and retry after the key is saved.
checkAuthRequired().then(required => {
    if (required && !getApiKey()) requestApiKey({ force: true });
});
// Reload what the user is looking at once a (new) key is saved, since
// requests made before it may have failed with 401.
onApiKeyChange(() => {
    if (!getApiKey()) return;
    dashboard.refreshInstances();
    PAGE_REFRESH[state.currentPage]?.();
});
connectWS();
settings.loadSettings();
dashboard.loadDashboardConfig();
dashboard.refreshInstances();
dashboard.refreshActivityTable();
visualizer.initVizDivider();
setInterval(updateClock, 1000);
updateClock();

setInterval(() => {
    if (state.currentPage === 'control' || state.currentPage === 'dashboard') {
        dashboard.refreshInstances();
        if (state.currentPage === 'control') control.refreshControlCenter();
        if (state.currentPage === 'dashboard') dashboard.refreshActivityTable();
    }
    // Incremental catch-up: appends only unseen records and respects pause.
    if (state.currentPage === 'logs') logs.syncLogs();
}, 10000);

navigateFromHash();
