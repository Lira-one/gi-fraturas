"""Paleta de cores, branding "GI" e componentes visuais (HTML/CSS) reutilizados
pelas telas do app.

Quando a cor definitiva da marca chegar, troque apenas as constantes abaixo
(e o [theme] de .streamlit/config.toml) — nenhum outro arquivo precisa mudar.
Todo o CSS é responsivo: abaixo de 640px (celular) os cartões empilham, as
fontes diminuem e elementos que não fazem sentido no celular (ex.: o próprio
QR code de acesso) são ocultados.
"""
from html import escape

import streamlit as st

PRIMARY_COLOR = "#0F6E5E"      # teal escuro clínico (provisório)
PRIMARY_DARK = "#0A4F43"
ACCENT_COLOR = "#2FA98C"
BACKGROUND_COLOR = "#F4F7F6"
SURFACE_COLOR = "#FFFFFF"
TEXT_COLOR = "#0B2B26"
MUTED_TEXT_COLOR = "#4B6B64"
BORDER_COLOR = "#D7E7E3"
# Cores de status: usadas só no resultado da análise e sempre com ícone +
# rótulo junto, para não depender apenas da cor (daltonismo).
DANGER_COLOR = "#D03B3B"       # status "critical" -> indicador "fraturado"
SUCCESS_COLOR = "#0CA30C"      # status "good" -> indicador "não fraturado"
SUCCESS_TEXT_COLOR = "#006300"

# Cores das séries nos gráficos (Treino × Validação, Sem fratura × Fraturado).
# Azul e laranja continuam distinguíveis para daltônicos.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
CHART_SURFACE = "#fcfcfb"
CHART_GRID = "#e1e0d9"
CHART_AXIS = "#c3c2b7"
CHART_MUTED = "#898781"
CHART_INK = "#0b0b0b"
CHART_INK_2 = "#52514e"

# Escala azul (claro -> escuro) da matriz de confusão.
SEQUENTIAL_BLUE_SCALE = [
    [0.0, "#cde2fb"],
    [0.33, "#6da7ec"],
    [0.66, "#2a78d6"],
    [1.0, "#0d366b"],
]

APP_NAME = "GI"
APP_TAGLINE = "Apoio ao Diagnóstico de Fraturas Ósseas"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def compact_html(html: str) -> str:
    """Junta o HTML numa linha só: no st.markdown, linhas indentadas após uma
    linha em branco viram bloco de código em vez de HTML."""
    return " ".join(line.strip() for line in html.splitlines() if line.strip())


def render_html(html: str):
    st.markdown(compact_html(html), unsafe_allow_html=True)


def pct(value: float, decimals: int = 1) -> str:
    """0.8435 -> '84,3%' (formato brasileiro)."""
    return f"{value * 100:.{decimals}f}%".replace(".", ",")


def num(value: float, decimals: int = 0) -> str:
    """1329 -> '1.329'; 0.9554 (decimals=3) -> '0,955'."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def inject_global_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .block-container, [data-testid="stMainBlockContainer"] {{
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}
        .stAppDeployButton, footer {{ display: none !important; }}

        /* ---------- Cabeçalho de página ---------- */
        .gi-page-header {{ margin: 0 0 1.25rem 0; }}
        .gi-eyebrow {{
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: {ACCENT_COLOR}; margin-bottom: 0.25rem;
        }}
        .gi-page-title {{
            font-size: 2rem; font-weight: 700; color: {TEXT_COLOR};
            line-height: 1.15; margin: 0;
        }}
        .gi-page-subtitle {{ color: {MUTED_TEXT_COLOR}; font-size: 1rem; margin-top: 0.35rem; }}

        /* ---------- Cartões ---------- */
        .gi-card {{
            background: {SURFACE_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 1px 2px rgba(11, 43, 38, 0.04), 0 4px 16px rgba(11, 43, 38, 0.04);
        }}
        .gi-section-title {{
            font-size: 1.05rem; font-weight: 650; color: {TEXT_COLOR}; margin: 0 0 0.15rem 0;
        }}
        .gi-section-caption {{ font-size: 0.88rem; color: {MUTED_TEXT_COLOR}; margin-bottom: 0.75rem; }}

        /* ---------- KPIs (grade que se adapta à largura) ---------- */
        .gi-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.75rem;
            margin: 0.25rem 0 0.75rem 0;
        }}
        .gi-kpi {{
            background: {SURFACE_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }}
        .gi-kpi-label {{ font-size: 0.8rem; color: {MUTED_TEXT_COLOR}; font-weight: 500; }}
        .gi-kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {TEXT_COLOR}; line-height: 1.2; margin-top: 0.15rem; }}
        .gi-kpi-hint {{ font-size: 0.76rem; color: {MUTED_TEXT_COLOR}; margin-top: 0.2rem; line-height: 1.35; }}
        .gi-kpi-highlight {{ border-color: {ACCENT_COLOR}; box-shadow: inset 3px 0 0 {ACCENT_COLOR}; }}

        /* ---------- Resultado da análise ---------- */
        .gi-result {{
            border-radius: 14px; padding: 1.2rem 1.35rem;
            border: 1px solid {BORDER_COLOR}; background: {SURFACE_COLOR};
        }}
        .gi-result-fractured {{ border-color: {DANGER_COLOR}; box-shadow: inset 5px 0 0 {DANGER_COLOR}; }}
        .gi-result-ok {{ border-color: {SUCCESS_COLOR}; box-shadow: inset 5px 0 0 {SUCCESS_COLOR}; }}
        .gi-result-eyebrow {{ font-size: 0.75rem; letter-spacing: 0.08em; font-weight: 600; color: {MUTED_TEXT_COLOR}; }}
        .gi-result-title {{ display:flex; align-items:center; gap:0.55rem; font-size: 1.75rem; font-weight: 700; margin: 0.3rem 0 0.2rem; color: {TEXT_COLOR}; }}
        .gi-badge {{
            display:inline-flex; align-items:center; justify-content:center;
            width: 2rem; height: 2rem; border-radius: 999px; color: #fff; font-size: 1.05rem; flex: none;
        }}
        .gi-badge-fractured {{ background: {DANGER_COLOR}; }}
        .gi-badge-ok {{ background: {SUCCESS_COLOR}; }}

        .gi-prob {{ margin-top: 1rem; }}
        .gi-prob-head {{ display:flex; justify-content:space-between; font-size: 0.85rem; color: {MUTED_TEXT_COLOR}; }}
        .gi-prob-head strong {{ color: {TEXT_COLOR}; font-size: 1rem; }}
        .gi-prob-track {{
            position: relative; height: 12px; border-radius: 999px;
            background: #E6EEEC; margin: 0.45rem 0 0.3rem; overflow: visible;
        }}
        .gi-prob-fill {{ position:absolute; left:0; top:0; bottom:0; border-radius: 999px; }}
        .gi-prob-threshold {{
            position:absolute; top:-5px; bottom:-5px; width:2px; background:{TEXT_COLOR}; opacity:0.55;
        }}
        .gi-prob-scale {{ display:flex; justify-content:space-between; font-size: 0.72rem; color: {MUTED_TEXT_COLOR}; }}

        .gi-facts {{ display:grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 1rem; }}
        .gi-fact {{ background: {BACKGROUND_COLOR}; border-radius: 10px; padding: 0.6rem 0.75rem; }}
        .gi-fact-label {{ font-size: 0.72rem; color: {MUTED_TEXT_COLOR}; }}
        .gi-fact-value {{ font-size: 0.95rem; font-weight: 600; color: {TEXT_COLOR}; word-break: break-word; }}

        /* ---------- Avisos ---------- */
        .gi-disclaimer {{
            display:flex; gap:0.7rem; align-items:flex-start;
            border: 1px solid #CBE6DF; background-color: #EAF6F3;
            padding: 0.8rem 1rem; border-radius: 12px; color: {TEXT_COLOR}; font-size: 0.9rem; line-height: 1.45;
        }}
        .gi-disclaimer-icon {{ font-size: 1.15rem; line-height: 1.2; }}

        /* ---------- Login / hero ---------- */
        .gi-hero {{
            background: linear-gradient(150deg, {PRIMARY_DARK} 0%, {PRIMARY_COLOR} 60%, #178A75 100%);
            color: #fff; border-radius: 18px; padding: 2rem 2rem 1.6rem;
            box-shadow: 0 10px 30px rgba(10, 79, 67, 0.18);
        }}
        .gi-hero h1 {{ color:#fff !important; font-size: 1.9rem; line-height: 1.15; margin: 1.1rem 0 0.6rem; padding: 0; }}
        .gi-hero p {{ color: rgba(255,255,255,0.86); font-size: 0.98rem; margin: 0; }}
        .gi-hero-features {{ list-style:none; padding:0; margin: 1.3rem 0 0; display:grid; gap:0.7rem; }}
        .gi-hero-features li {{ display:flex; gap:0.65rem; align-items:flex-start; font-size:0.92rem; color: rgba(255,255,255,0.92); }}
        .gi-hero-features .gi-dot {{
            flex:none; width:1.6rem; height:1.6rem; border-radius:8px; background: rgba(255,255,255,0.14);
            display:inline-flex; align-items:center; justify-content:center; font-size:0.9rem;
        }}
        .gi-hero-stats {{ display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:1.4rem; }}
        .gi-hero-stat {{ background: rgba(255,255,255,0.12); border-radius: 10px; padding: 0.45rem 0.75rem; font-size: 0.8rem; }}
        .gi-hero-stat b {{ font-size: 1rem; display:block; }}

        /* ---------- QR code de acesso pelo celular ---------- */
        .gi-qr {{
            display:flex; gap:1rem; align-items:center;
            background: {SURFACE_COLOR}; border: 1px dashed {ACCENT_COLOR};
            border-radius: 14px; padding: 0.9rem 1rem; margin-top: 1rem;
        }}
        .gi-qr img {{ width: 112px; height: 112px; border-radius: 8px; flex: none; image-rendering: pixelated; }}
        .gi-qr-title {{ font-weight: 650; color: {TEXT_COLOR}; font-size: 0.95rem; }}
        .gi-qr-text {{ font-size: 0.82rem; color: {MUTED_TEXT_COLOR}; margin-top: 0.2rem; line-height: 1.4; }}
        .gi-qr-url {{
            display:inline-block; margin-top: 0.4rem; font-family: ui-monospace, Consolas, monospace;
            font-size: 0.8rem; background: {BACKGROUND_COLOR}; color: {PRIMARY_DARK};
            padding: 0.15rem 0.45rem; border-radius: 6px; word-break: break-all;
        }}
        .gi-qr-sidebar {{ flex-direction: column; text-align: center; background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.35); }}
        .gi-qr-sidebar img {{ width: 150px; height: 150px; background:#fff; padding:6px; }}
        .gi-qr-sidebar .gi-qr-title, .gi-qr-sidebar .gi-qr-text {{ color: #fff; }}
        .gi-qr-sidebar .gi-qr-url {{ background: rgba(255,255,255,0.12); color:#fff; }}

        /* ---------- Sidebar ---------- */
        .gi-user {{ display:flex; gap:0.7rem; align-items:center; padding: 0.4rem 0 0.2rem; }}
        .gi-avatar {{
            width: 2.4rem; height: 2.4rem; border-radius: 999px; background: {ACCENT_COLOR}; color:#fff;
            display:inline-flex; align-items:center; justify-content:center; font-weight:700; flex:none;
        }}
        .gi-user-name {{ font-weight: 650; font-size: 0.95rem; line-height: 1.2; }}
        .gi-user-role {{ font-size: 0.78rem; opacity: 0.8; }}

        /* ---------- Gráficos ---------- */
        [data-testid="stPlotlyChart"] {{
            background: {CHART_SURFACE}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 0.5rem;
        }}

        /* ---------- Celular ---------- */
        @media (max-width: 640px) {{
            .block-container, [data-testid="stMainBlockContainer"] {{
                padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem;
            }}
            .gi-page-title {{ font-size: 1.5rem; }}
            .gi-page-subtitle {{ font-size: 0.92rem; }}
            .gi-kpi-grid {{ grid-template-columns: 1fr 1fr; gap: 0.55rem; }}
            .gi-kpi {{ padding: 0.7rem 0.75rem; }}
            .gi-kpi-value {{ font-size: 1.35rem; }}
            .gi-result-title {{ font-size: 1.4rem; }}
            .gi-hero {{ padding: 1.4rem 1.2rem 1.2rem; border-radius: 14px; }}
            .gi-hero h1 {{ font-size: 1.45rem; margin-top: 0.8rem; }}
            .gi-hero-features {{ display: none; }}
            .gi-hide-mobile {{ display: none !important; }}
            .gi-facts {{ gap: 0.45rem; }}
            .gi-fact {{ padding: 0.5rem 0.6rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def logo_svg(box: int = 44, light: bool = False) -> str:
    fill = "rgba(255,255,255,0.14)" if light else PRIMARY_COLOR
    return f"""
    <svg width="{box}" height="{box}" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <rect x="2" y="2" width="68" height="68" rx="18" fill="{fill}" stroke="{ACCENT_COLOR}" stroke-width="2"/>
        <text x="50%" y="56%" text-anchor="middle" dominant-baseline="middle"
              font-family="Inter, Arial, Helvetica, sans-serif" font-weight="700"
              font-size="30" fill="#FFFFFF">GI</text>
    </svg>
    """


def render_logo(size: str = "lg"):
    """Wordmark "GI" em SVG inline, com tagline opcional no tamanho grande."""
    box = 56 if size == "lg" else 38
    tagline = (
        f'<div style="font-size:0.82rem; opacity:0.8;">{APP_TAGLINE}</div>' if size == "lg" else ""
    )
    render_html(
        f"""
        <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
            {logo_svg(box)}
            <div>
                <div style="font-size:{'1.35rem' if size == 'lg' else '1.1rem'}; font-weight:700; line-height:1.1;">{APP_NAME}</div>
                {tagline}
            </div>
        </div>
        """
    )


def page_header(title: str, subtitle: str = "", eyebrow: str = ""):
    eyebrow_html = f'<div class="gi-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f'<div class="gi-page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="gi-page-header">{eyebrow_html}<h1 class="gi-page-title">{escape(title)}</h1>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: str = ""):
    caption_html = f'<div class="gi-section-caption">{caption}</div>' if caption else ""
    st.markdown(f'<div class="gi-section-title">{escape(title)}</div>{caption_html}', unsafe_allow_html=True)


def kpi_grid(items):
    """items: lista de dicts {label, value, hint?, highlight?}. Grade CSS que
    se reorganiza sozinha (5 colunas no desktop, 2 no celular)."""
    cards = []
    for item in items:
        cls = "gi-kpi gi-kpi-highlight" if item.get("highlight") else "gi-kpi"
        hint = f'<div class="gi-kpi-hint">{item["hint"]}</div>' if item.get("hint") else ""
        cards.append(
            f'<div class="{cls}"><div class="gi-kpi-label">{escape(item["label"])}</div>'
            f'<div class="gi-kpi-value">{escape(str(item["value"]))}</div>{hint}</div>'
        )
    st.markdown(f'<div class="gi-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def disclaimer(html: str, icon: str = "ℹ️"):
    st.markdown(
        f'<div class="gi-disclaimer"><span class="gi-disclaimer-icon">{icon}</span><div>{html}</div></div>',
        unsafe_allow_html=True,
    )


def base_chart_layout(height: int = 320, **overrides) -> dict:
    """Layout Plotly padrão: superfície/tinta da paleta validada, grade
    recessiva, legenda horizontal no topo (cabe no celular) e hover unificado."""
    layout = dict(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        plot_bgcolor=CHART_SURFACE,
        paper_bgcolor=CHART_SURFACE,
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=CHART_INK_2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=CHART_INK_2)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=CHART_AXIS, font=dict(color=CHART_INK)),
        xaxis=dict(gridcolor=CHART_GRID, linecolor=CHART_AXIS, zeroline=False, tickfont=dict(color=CHART_MUTED)),
        yaxis=dict(gridcolor=CHART_GRID, linecolor=CHART_AXIS, zeroline=False, tickfont=dict(color=CHART_MUTED)),
    )
    for axis in ("xaxis", "yaxis", "legend"):
        if axis in overrides:
            layout[axis] = {**layout[axis], **overrides.pop(axis)}
    layout.update(overrides)
    return layout
