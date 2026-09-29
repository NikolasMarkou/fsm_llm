# pages

The screens of the FSM-LLM Monitor web dashboard. Each file renders one page (or modal) and handles the buttons on it.

## What it is for

The FSM-LLM Monitor is a browser dashboard for the FSM-LLM framework. From it you can launch FSM chatbots, agents, and workflows, chat with running conversations, watch events and logs as they happen, draw state diagrams, and build new definitions with an AI assistant. This folder holds the code for each of those screens. The pages build HTML strings, fetch data from the monitor server's REST API, and react to live updates pushed over a WebSocket.

## How it works

`static/app.js` imports every page module, wires them together, and routes clicks to them. Pages do not import `app.js`; instead `app.js` hands each page the functions it needs (such as `showPage`) through a `setDeps(...)` call. Live data arrives through the WebSocket manager, which calls page functions like `updateMetrics` or `appendLogs`.

| Page | What you do there |
| --- | --- |
| Dashboard | See counters, the event feed, a grid of running instances, and an activity table |
| Control Center | A filterable table of all instances; click one to open a side drawer with details, events, and actions |
| Conversations | Inside the drawer: read an FSM conversation's state, context, last LLM calls, and chat with it |
| Launch modal | Start an FSM (from a preset or pasted JSON), a workflow preset, or an agent with stub tools |
| Visualizer | Paste or pick an FSM definition and see it drawn as a graph |
| Builder | Chat with the meta-builder agent to create an FSM, workflow, or agent definition, then launch it |
| Logs | Live log stream with level filters, search, pause, and jump-to-latest |
| Settings | Monitor refresh rate, buffer sizes, log level, and system info |

## Files

- `dashboard.js` - metrics cards, event feed, instance grid, activity table, optional custom dashboard panels and alerts.
- `control.js` - unified instance table, detail drawer for FSM, workflow, and agent instances, agent trace view, cancel and destroy actions.
- `conversations.js` - conversation detail view and chat box inside the drawer; typing indicator reused by the builder.
- `launch.js` - launch modal for FSMs, workflows, and agents, including 15 ready-made stub tool templates.
- `visualizer.js` - FSM, agent, and workflow graph drawing, preset dropdown, resizable split pane.
- `builder.js` - chat session with the meta-builder, internal state panel, result summary, copy, download, and launch.
- `logs.js` - log stream with pause buffer, level pills, search highlight, sidebar error badge.
- `settings.js` - load, save, and reset monitor configuration.

## How to use it

Start the dashboard and open it in a browser:

```bash
fsm-llm-monitor
# then open http://127.0.0.1:8420
```

Every page is reachable from the sidebar. To launch something, open the Launch modal, pick a preset such as a basic greeting FSM, and press Launch FSM. The Control Center opens with the new conversation in the drawer.

## Things to know

- Agents that act by calling tools (ReAct, Reflexion, Plan-Execute, REWOO, ADaPT) need at least one tool at launch. The monitor only offers stub tools that return a fixed string you type in.
- The drawer re-fetches its content every 2 seconds while it is open on the Control Center.
- The Logs page only draws new entries while you are on it. On other pages it still updates the error badge and counts, and it reloads the latest 500 entries from the server when you come back.
- While logs are paused, up to 5,000 new entries are buffered; older ones are dropped. The visible stream keeps at most 1,000 entries.
- Custom dashboard panels only appear when a dashboard config has been set on the server.
