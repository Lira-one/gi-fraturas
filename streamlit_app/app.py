"""Ponto de entrada do app GI: tela de login seguida de roteamento por papel
(Radiologista / Gerente-Chefe Clínico).

Rodar a partir da raiz do projeto com:
    streamlit run streamlit_app/app.py

O app fica acessível na rede local (ex.: celular na mesma Wi-Fi) pelo QR code
exibido na tela de login e na barra lateral — ver mobile_access.py.
"""
import sys
from pathlib import Path

# Garante que streamlit_app/ esteja no sys.path independente de como o script
# é executado (streamlit run, AppTest, etc.) para que os imports locais abaixo
# sempre funcionem.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from auth import ROLE_LABELS, authenticate
from mobile_access import render_mobile_access
from model_metrics import load_metrics
from theme import APP_NAME, APP_TAGLINE, inject_global_css, logo_svg, num, pct, render_html, render_logo

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_global_css()

if "user" not in st.session_state:
    st.session_state.user = None


def _render_hero():
    metrics = load_metrics()
    stats = ""
    if metrics:
        stats = f"""
        <div class="gi-hero-stats">
            <div class="gi-hero-stat"><b>{pct(metrics['recall'], 0)}</b>sensibilidade p/ fraturas</div>
            <div class="gi-hero-stat"><b>{num(metrics.get('auc', 0), 2)}</b>AUC no teste</div>
            <div class="gi-hero-stat"><b>{num(metrics['test_size'])}</b>radiografias de teste</div>
        </div>
        """
    render_html(
        f"""
        <div class="gi-hero">
            <div style="display:flex; align-items:center; gap:0.7rem;">
                {logo_svg(46, light=True)}
                <div style="font-weight:700; font-size:1.15rem;">{APP_NAME}
                    <div style="font-weight:400; font-size:0.8rem; opacity:0.8;">{APP_TAGLINE}</div>
                </div>
            </div>
            <h1>Uma segunda leitura da radiografia, em segundos.</h1>
            <p>Rede neural (MobileNetV2) treinada para sinalizar possíveis fraturas e mostrar
            onde ela concentrou atenção. O laudo continua com o radiologista.</p>
            <ul class="gi-hero-features">
                <li><span class="gi-dot">🩻</span><span><b>Análise imediata</b> — envie JPG/PNG e receba a classificação com a probabilidade estimada.</span></li>
                <li><span class="gi-dot">📍</span><span><b>Mapa de atenção (Grad-CAM)</b> — destaca a região que mais influenciou o modelo.</span></li>
                <li><span class="gi-dot">📊</span><span><b>Painel gerencial</b> — métricas no teste, curvas ROC/PR e atividade da equipe.</span></li>
            </ul>
            {stats}
        </div>
        """
    )


def render_login():
    st.write("")
    col_hero, col_form = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with col_hero:
        _render_hero()
    with col_form:
        with st.container(border=True):
            st.markdown("#### Entrar na plataforma")
            st.caption("Use suas credenciais institucionais.")
            with st.form("login_form", border=False):
                username = st.text_input("Usuário", placeholder="ex.: dra.ana")
                password = st.text_input("Senha", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Entrar", type="primary", width="stretch")
            if submitted:
                user = authenticate(username.strip(), password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")
            with st.expander("Usuários de demonstração"):
                st.markdown(
                    "| Perfil | Usuário | Senha |\n|---|---|---|\n"
                    "| Radiologista | `dra.ana` | `raiox2026` |\n"
                    "| Gerente / Chefe Clínico | `gerente.carlos` | `gestao2026` |"
                )
        render_mobile_access("card")


def render_app():
    user = st.session_state.user
    initials = "".join(p[0] for p in user["display_name"].replace(".", " ").split()[:2]).upper()
    with st.sidebar:
        render_logo(size="sm")
        st.divider()
        render_html(
            f"""
            <div class="gi-user">
                <span class="gi-avatar">{initials}</span>
                <div>
                    <div class="gi-user-name">{user['display_name']}</div>
                    <div class="gi-user-role">{ROLE_LABELS.get(user['role'], user['role'])}</div>
                </div>
            </div>
            """
        )
        st.write("")
        if st.button("Sair", icon=":material/logout:", width="stretch"):
            st.session_state.user = None
            st.rerun()

        if user["role"] == "gerente":
            from user_admin import render_user_admin
            render_user_admin()

        st.divider()
        render_mobile_access("sidebar")

    if user["role"] == "radiologista":
        from pages_logic import radiologista as page
    else:
        from pages_logic import gerente as page
    page.render()


if st.session_state.user is None:
    render_login()
else:
    render_app()
