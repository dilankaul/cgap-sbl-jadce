"""Dashboard presentation: local vector icons, task cards, and a shared theme."""
from html import escape

# Small inline SVGs keep task/status symbols sharp and available offline.
PATHS = {
    'activity': '<circle cx="12" cy="12" r="2"/><path d="M8 8a6 6 0 0 0 0 8m8-8a6 6 0 0 1 0 8M5 5a10 10 0 0 0 0 14m14-14a10 10 0 0 1 0 14"/>',
    'communication': '<path d="M3 12h3l3-8 6 16 3-8h3"/>',
    'spatial_dependence': '<circle cx="5" cy="5" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="19" r="2"/><path d="m7 5 10 1M6 7l5 10m7-9-5 9"/>',
    'omp': '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    'cosamp': '<path d="m12 3 10 5-10 5L2 8l10-5zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
    'sbl': '<path d="M3 18c4 0 3-12 9-12s5 12 9 12M3 21h18"/>',
    'anc': '<path d="m12 3 8 4v5c0 5-8 9-8 9s-8-4-8-9V7l8-4zM8 12l3 3 5-6"/>',
    'comparison': '<path d="M4 20V10m8 10V4m8 16v-7M2 20h20"/>',
    'convergence': '<path d="M3 3v18h18M6 6c2 0 2 8 6 10s6 1 9 1"/>',
    'tune': '<path d="M5 3v5m0 4v9M12 3v10m0 4v4M19 3v2m0 4v12M2 8h6m1 9h6m1-12h6"/>',
    'kappa_calibration': '<path d="M5 3v5m0 4v9M12 3v10m0 4v4M19 3v2m0 4v12M2 8h6m1 9h6m1-12h6"/>',
    'arrow': '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    'completed': '<path d="m5 12 4 4L19 6"/>',
    'ready': '<circle cx="12" cy="12" r="5"/>',
    'running': '<path d="m8 5 10 7-10 7V5z"/>',
    'queued': '<path d="M5 3h14M5 21h14M7 3v4l5 5-5 5v4m10-18v4l-5 5 5 5v4"/>',
    'paused': '<path d="M8 5v14M16 5v14"/>',
    'stopped': '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    'partial': '<circle cx="12" cy="12" r="9"/><path d="M12 3v18"/>',
    'failed': '<circle cx="12" cy="12" r="9"/><path d="M12 7v6m0 3v.1"/>',
    'blocked': '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    'incompatible': '<path d="m12 3 10 18H2L12 3zM12 9v5m0 3v.1"/>',
    'historical': '<path d="M3 11a9 9 0 1 1 3 8M3 4v7h7m2-5v6l4 2"/>',
}


def icon(name):
    name = 'failed' if name == 'interrupted' else name
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{PATHS.get(name, PATHS["activity"])}</svg>')


def card_markup(title, task, state, action='run', compact=False):
    task_icon = 'tune' if action == 'tune' else task
    return (f'<div class="workflow-card {"compact" if compact else "wide"}">'
            f'<div class="card-icon">{icon(task_icon)}</div>'
            f'<div class="card-copy"><div class="card-title">{escape(title)}</div>'
            f'<div class="card-detail">{escape(state.summary)}</div></div>'
            f'<span class="status-badge status-{state.status}">{icon(state.status)}{escape(state.status.capitalize())}</span>'
            f'<span class="card-arrow">{icon("arrow")}</span></div>')


def section_heading(st, title, subtitle=''):
    st.markdown(f'<div class="section-heading"><h2>{escape(title)}</h2><span>{escape(subtitle)}</span></div>', unsafe_allow_html=True)


def apply_theme(st):
    st.markdown(STYLES, unsafe_allow_html=True)


STYLES = '''<style>
:root { --lab-ink:#172b3a; --lab-muted:#64748b; --lab-accent:#0f766e; --lab-border:#e2e8f0; }
[data-testid="stAppViewContainer"] { background:#f5f7fa; color:var(--lab-ink); }
.block-container { padding-top:4rem; padding-bottom:3rem; max-width:1250px; }
[data-testid="stHeader"] { background:rgba(245,247,250,.92); }
[data-testid="stAppDeployButton"] { display:none; }
h1,h2,h3 { color:var(--lab-ink); letter-spacing:-.035em; }
h1 { font-size:2.25rem!important; font-weight:700!important; }
h2 { font-size:1.2rem!important; } h3 { font-size:1.1rem!important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--lab-muted)!important; }
[data-testid="stSidebar"] { background:#eef2f6; border-right:1px solid var(--lab-border); }
[data-testid="stSidebarUserContent"] { padding-top:2rem; }
.lab-brand { display:flex; gap:12px; align-items:center; margin-bottom:30px; }
.lab-brand-mark { color:#fff; background:var(--lab-ink); padding:10px; border-radius:13px; width:44px; height:44px; }
.lab-brand-mark svg { width:24px; height:24px; }
.lab-brand strong { font-size:1.12rem; letter-spacing:-.025em; }
.lab-brand small { display:block; color:var(--lab-muted); font-size:12px; margin-top:3px; }
.sidebar-label { font-size:10px; letter-spacing:.13em; font-weight:700; color:var(--lab-muted); margin:20px 0 8px; }
.setup-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin:20px 0 14px; }
.setup-summary div { padding:10px 8px; background:#fff; border:1px solid var(--lab-border); border-radius:10px; }
.setup-summary strong { display:block; font-size:18px; font-weight:650; }
.setup-summary span { color:var(--lab-muted); font-size:11px; }
.eyebrow { color:var(--lab-accent); letter-spacing:.13em; font-size:11px; font-weight:700; margin-bottom:6px; }
.section-heading { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:24px 0 16px; }
.section-heading h2 { margin:0; padding:0; font-weight:650; }
.section-heading > span { color:var(--lab-muted); font-size:12px; text-align:right; }
[class*="st-key-card_"] { position:relative; border:1px solid var(--lab-border); border-radius:16px;
    background:#fff; box-shadow:0 2px 4px rgba(15,23,42,.025); overflow:hidden; gap:0!important;
    transition:box-shadow .18s,border-color .18s,transform .18s; }
/* Stretch each card to the height of its column's row. */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-card_"]) { flex:1; }
[data-testid="stColumn"] [class*="st-key-card_"] { justify-content:center; }
/* Keep the card's visible bounds aligned with its markdown content. */
[class*="st-key-card_"] [data-testid="stMarkdownContainer"] { margin-bottom:0!important; }
[class*="st-key-card_"]:hover { border-color:#7bb8b0; box-shadow:0 6px 22px rgba(15,23,42,.07); transform:translateY(-2px); }
.workflow-card { display:flex; align-items:center; align-content:center; gap:18px; padding:22px 24px; min-height:96px; pointer-events:none; }
.card-icon { display:flex; align-items:center; justify-content:center; width:46px; height:46px; flex-shrink:0;
    background:#edf6f5; color:var(--lab-accent); border-radius:13px; }
.card-icon svg { width:25px; height:25px; }
.card-copy { flex:1; min-width:0; }
.card-title { font-size:16px; font-weight:650; line-height:1.35; letter-spacing:-.018em; }
.card-detail { color:var(--lab-muted); font-size:12px; line-height:1.6; margin-top:5px; overflow-wrap:anywhere; }
.card-arrow { color:#94a3b8; width:19px; flex-shrink:0; } .card-arrow svg { width:19px; height:19px; }
.status-badge { display:inline-flex; align-items:center; gap:5px; padding:5px 9px; border-radius:20px;
    font-size:11px; font-weight:600; color:#64748b; background:#f1f5f9; white-space:nowrap; }
.status-badge svg { width:13px; height:13px; }
.status-completed { color:#047857; background:#ecfdf5; }
.status-running,.status-queued { color:#0369a1; background:#e0f2fe; }
.status-paused,.status-partial { color:#a16207; background:#fef9c3; }
.status-failed,.status-interrupted { color:#be123c; background:#fff1f2; }
.status-incompatible { color:#9a3412; background:#ffedd5; }
.status-blocked { color:#b45309; background:#fff7ed; }
.compact { flex-wrap:wrap; }
.compact .card-copy { flex-basis:100%; order:2; }
.compact .status-badge { margin-left:auto; }
.compact .card-arrow { display:none; }
[class*="st-key-card_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
    position:absolute; inset:0; z-index:2; height:100%; }
[class*="st-key-card_"] [data-testid="stButton"] { height:100%; }
[class*="st-key-card_"] button { width:100%!important; height:100%!important; background:transparent!important;
    border:0!important; border-radius:16px; box-shadow:none!important; }
[class*="st-key-card_"] button p { opacity:0; }
[class*="st-key-card_"] button:focus-visible { outline:3px solid #0d9488!important; outline-offset:-3px; }
[class*="st-key-card_"] [data-testid="stProgress"] { padding:0 22px 14px; }
[data-testid="stBaseButton-primary"] { background:#0f766e!important; border-color:#0f766e!important; color:#fff!important; }
[data-testid="stBaseButton-primary"]:hover { background:#115e59!important; border-color:#115e59!important; }
button { border-radius:10px!important; transition:background .15s,border-color .15s; }
[data-testid="stSidebar"] button { justify-content:flex-start; padding:10px 14px; border:0; background:transparent; }
[data-testid="stSidebar"] button:hover { background:#e2e8f0; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background:#dcefea!important; color:#0f766e!important; border:0!important; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--lab-border); padding:16px; border-radius:12px; }
[data-testid="stExpander"] { border-color:var(--lab-border); background:#fff; border-radius:12px; }
[data-testid="stForm"] { background:#fff; border-color:var(--lab-border); border-radius:16px; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid var(--lab-border); border-radius:16px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stWidgetLabel"] p { font-size:13px; font-weight:550; }
[data-baseweb="select"] > div { border-radius:10px; }
[data-testid="stNumberInputContainer"],[data-testid="stTextInputRootElement"] { border-radius:10px; }
hr { border-color:var(--lab-border)!important; }
@media (max-width:800px) {
    .block-container { padding:4rem 1rem 1.3rem; }
}
/* Respond to the workspace left after the sidebar and page padding. */
.block-container { container:workspace / inline-size; }
[class*="st-key-card_"] { container:taskcard / inline-size; }
.card-title { overflow-wrap:normal; word-break:normal; }
@container workspace (max-width:960px) {
    [data-testid="stHorizontalBlock"]:has(.compact) > [data-testid="stColumn"] {
        flex:1 1 calc(50% - 16px)!important; min-width:0!important; width:auto!important;
    }
}
@container workspace (max-width:760px) {
    [data-testid="stHorizontalBlock"]:has(.wide) > [data-testid="stColumn"] {
        flex:1 1 100%!important; min-width:0!important; width:auto!important;
    }
    .section-heading > span { display:none; }
}
@container workspace (max-width:480px) {
    [data-testid="stHorizontalBlock"]:has(.compact) > [data-testid="stColumn"] {
        flex:1 1 100%!important;
    }
    /* Single-column algorithms use the same row layout as preparation cards. */
    .compact { flex-wrap:nowrap; }
    .compact .card-copy { flex-basis:auto; order:0; }
    .compact .status-badge { margin-left:0; }
    .compact .card-arrow { display:block; }
    @container taskcard (max-width:450px) {
        .compact { display:grid; grid-template-columns:46px minmax(0,1fr); gap:12px; padding:18px; }
        .compact .status-badge { grid-column:2; justify-self:start; }
        .compact .card-arrow { display:none; }
    }
}
@container taskcard (max-width:450px) {
    .wide { display:grid; grid-template-columns:46px minmax(0,1fr); gap:12px; padding:18px; }
    .wide .status-badge { grid-column:2; justify-self:start; }
    .wide .card-arrow { display:none; }
}
@media (prefers-reduced-motion:reduce) {
    [class*="st-key-card_"] { transition:none; }
    [class*="st-key-card_"]:hover { transform:none; }
}
</style>'''
