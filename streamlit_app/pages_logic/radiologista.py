"""Painel do Médico(a) Radiologista: upload de radiografia e predição de apoio."""
import hashlib
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from gradcam import compute_gradcam_heatmap, overlay_heatmap_on_image
from model_utils import THRESHOLD, load_model, model_is_available, predict_from_array, prepare_image_array
from theme import (
    DANGER_COLOR,
    SUCCESS_COLOR,
    disclaimer,
    num,
    page_header,
    pct,
    render_html,
    section_title,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "Imagens_Teste"
SAMPLE_IMAGES = ["Imagem_teste.jpg", "New_Image.png"]

# Faixa em torno do limiar em que o resultado é sinalizado como "limítrofe".
BORDERLINE_MARGIN = 0.15


# ---------------------------------------------------------------- estado ----

def _set_active_image(name: str, data: bytes, source: str):
    st.session_state.active_image = {"name": name, "bytes": data, "source": source}


def _on_upload():
    uploaded = st.session_state.get("uploader")
    if uploaded is not None:
        _set_active_image(uploaded.name, uploaded.getvalue(), "upload")
    elif st.session_state.get("active_image", {}).get("source") == "upload":
        st.session_state.active_image = None


def _analyze(name: str, data: bytes) -> dict:
    """Roda predição + Grad-CAM uma única vez por imagem (chave = hash do
    conteúdo) e registra o exame no histórico só na primeira vez — reruns do
    Streamlit (qualquer clique) não duplicam o registro."""
    analyses = st.session_state.setdefault("analyses", {})
    key = hashlib.sha1(data).hexdigest()
    if key in analyses:
        return analyses[key]

    image = Image.open(io.BytesIO(data))
    image.load()
    arr = prepare_image_array(image)
    with st.spinner("Analisando radiografia..."):
        result = predict_from_array(arr)
        heatmap = compute_gradcam_heatmap(load_model(), arr)
        color = DANGER_COLOR if result["is_fractured"] else SUCCESS_COLOR
        overlay = overlay_heatmap_on_image(image, heatmap, color_hex=color)

    analysis = {
        "name": name,
        "image": image.convert("RGB"),
        "overlay": overlay,
        "result": result,
        "size": image.size,
        "analyzed_at": datetime.now(),
    }
    analyses[key] = analysis
    st.session_state.setdefault("exam_history", []).append(
        {
            "timestamp": analysis["analyzed_at"].strftime("%d/%m/%Y %H:%M:%S"),
            "arquivo": name,
            "resultado": result["label"],
            "confianca": pct(result["confidence"]),
            "prob_fratura": result["raw_probability_fractured"],
        }
    )
    return analysis


# ----------------------------------------------------------- componentes ----

@st.cache_data(show_spinner=False)
def _thumbnail(data: bytes) -> Image.Image:
    """Miniatura quadrada (recorte central) para a galeria de exemplos."""
    return ImageOps.fit(Image.open(io.BytesIO(data)).convert("RGB"), (320, 240))


def _render_input():
    tab_upload, tab_samples = st.tabs([":material/upload: Enviar radiografia", ":material/photo_library: Imagens de exemplo"])
    with tab_upload:
        st.file_uploader(
            "Radiografia de antebraço (JPG ou PNG)",
            type=["jpg", "jpeg", "png"],
            key="uploader",
            on_change=_on_upload,
            help="No celular, você pode escolher da galeria ou tirar uma foto da imagem.",
        )
    with tab_samples:
        st.caption("Imagens do próprio projeto (pasta `Imagens_Teste`) para demonstrar o fluxo.")
        available = [n for n in SAMPLE_IMAGES if (SAMPLES_DIR / n).exists()]
        # 4 colunas fixas: miniaturas pequenas no desktop; no celular empilham.
        cols = st.columns(4)
        for col, sample_name in zip(cols, available):
            with col, st.container(border=True):
                data = (SAMPLES_DIR / sample_name).read_bytes()
                st.image(_thumbnail(data), width="stretch")
                st.caption(sample_name)
                if st.button("Analisar", key=f"sample_{sample_name}", icon=":material/play_arrow:", width="stretch"):
                    _set_active_image(sample_name, data, "sample")


def _probability_bar(prob: float, is_fractured: bool) -> str:
    color = DANGER_COLOR if is_fractured else SUCCESS_COLOR
    return f"""
    <div class="gi-prob">
        <div class="gi-prob-head"><span>Probabilidade estimada de fratura</span><strong>{pct(prob)}</strong></div>
        <div class="gi-prob-track" role="img" aria-label="Probabilidade de fratura {pct(prob)}, limiar {pct(THRESHOLD, 0)}">
            <div class="gi-prob-fill" style="width:{prob * 100:.1f}%; background:{color};"></div>
            <div class="gi-prob-threshold" style="left:calc({THRESHOLD * 100:.0f}% - 1px);" title="Limiar de decisão"></div>
        </div>
        <div class="gi-prob-scale"><span>0%</span><span>limiar {pct(THRESHOLD, 0)}</span><span>100%</span></div>
    </div>
    """


def _report_text(analysis: dict) -> str:
    r = analysis["result"]
    return "\n".join([
        "GI — Apoio ao Diagnóstico de Fraturas Ósseas",
        "Relatório de análise automática (apoio à decisão)",
        "=" * 52,
        f"Arquivo:                 {analysis['name']}",
        f"Data/hora da análise:    {analysis['analyzed_at'].strftime('%d/%m/%Y %H:%M:%S')}",
        f"Resultado do modelo:     {r['label']}",
        f"Prob. estimada fratura:  {pct(r['raw_probability_fractured'])}",
        f"Limiar de decisão:       {pct(THRESHOLD, 0)}",
        "Modelo:                  MobileNetV2 (Transfer Learning)",
        "",
        "AVISO: estimativa automática de um modelo experimental, sem calibração",
        "clínica. Não substitui o laudo do médico radiologista responsável.",
    ])


def _render_result(analysis: dict):
    result = analysis["result"]
    prob = result["raw_probability_fractured"]
    is_fractured = result["is_fractured"]

    st.write("")
    section_title("Resultado da análise", f"Arquivo <b>{analysis['name']}</b>")
    col_img, col_res = st.columns([1.1, 1], gap="large")

    with col_img:
        view = st.segmented_control(
            "Visualização",
            ["Original", "Mapa de atenção"],
            default="Mapa de atenção",
            key=f"view_{analysis['name']}",
            label_visibility="collapsed",
        )
        if view == "Original":
            st.image(analysis["image"], width="stretch")
        else:
            st.image(analysis["overlay"], width="stretch")
            st.caption("Grad-CAM: região que mais influenciou a predição (aproximada, grade 7×7).")

    with col_res:
        card_class = "gi-result-fractured" if is_fractured else "gi-result-ok"
        badge_class = "gi-badge-fractured" if is_fractured else "gi-badge-ok"
        icon = "!" if is_fractured else "✓"
        borderline = abs(prob - THRESHOLD) < BORDERLINE_MARGIN
        if borderline:
            reading = "⚠️ <b>Resultado limítrofe</b>: a probabilidade está próxima do limiar. Revise a imagem com atenção redobrada."
        elif is_fractured:
            reading = "Padrão compatível com fratura identificado. Verifique a área destacada no mapa de atenção."
        else:
            reading = "O modelo não identificou padrão de fratura nesta imagem."
        width, height = analysis["size"]
        render_html(
            f"""
            <div class="gi-result {card_class}">
                <div class="gi-result-eyebrow">CLASSIFICAÇÃO DO MODELO</div>
                <div class="gi-result-title"><span class="gi-badge {badge_class}">{icon}</span>{result['label']}</div>
                <div style="font-size:0.92rem; color:#4B6B64; line-height:1.45;">{reading}</div>
                {_probability_bar(prob, is_fractured)}
                <div class="gi-facts">
                    <div class="gi-fact"><div class="gi-fact-label">Confiança na classe</div><div class="gi-fact-value">{pct(result['confidence'])}</div></div>
                    <div class="gi-fact"><div class="gi-fact-label">Resolução original</div><div class="gi-fact-value">{num(width)} × {num(height)} px</div></div>
                    <div class="gi-fact"><div class="gi-fact-label">Analisado em</div><div class="gi-fact-value">{analysis['analyzed_at'].strftime('%d/%m/%Y %H:%M')}</div></div>
                    <div class="gi-fact"><div class="gi-fact-label">Modelo</div><div class="gi-fact-value">MobileNetV2</div></div>
                </div>
            </div>
            """
        )
        st.write("")
        st.download_button(
            "Baixar relatório (.txt)",
            data=_report_text(analysis),
            file_name=f"relatorio_{Path(analysis['name']).stem}.txt",
            mime="text/plain",
            icon=":material/download:",
            width="stretch",
        )

    with st.expander("Como interpretar este resultado"):
        st.markdown(
            "- **Probabilidade estimada** é a saída da função *sigmoid* do modelo. Ela **não é uma "
            "probabilidade clínica calibrada** — no conjunto de teste, o modelo tende a superestimar "
            "a chance de fratura na faixa intermediária (ver curva de calibração no painel gerencial).\n"
            "- **Mapa de atenção (Grad-CAM)** mostra *para onde o modelo olhou*, não uma segmentação "
            "da fratura. A área marcada é aproximada (grade 7×7 ampliada para a imagem).\n"
            f"- O **limiar de decisão** é {pct(THRESHOLD, 0)}: acima dele, a imagem é classificada como "
            "*Fraturado*. Resultados a menos de 15 pontos do limiar são sinalizados como limítrofes.\n"
            "- O modelo só conhece radiografias parecidas com as do dataset de treino: imagens de "
            "outros equipamentos, ossos ou projeções podem receber resultados sem relação confiável com a realidade."
        )


def _render_history():
    history = st.session_state.get("exam_history", [])
    if not history:
        return
    st.divider()
    section_title(
        "Exames analisados nesta sessão",
        "Histórico temporário (apenas desta sessão do navegador); alimenta o painel do Gerente/Chefe Clínico.",
    )
    df = pd.DataFrame(history).iloc[::-1]
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Data/hora"),
            "arquivo": st.column_config.TextColumn("Arquivo"),
            "resultado": st.column_config.TextColumn("Resultado"),
            "confianca": st.column_config.TextColumn("Confiança"),
            "prob_fratura": st.column_config.ProgressColumn(
                "Prob. fratura", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )


# ----------------------------------------------------------------- página ----

def render():
    page_header(
        "Análise de Radiografia",
        "Envie uma radiografia de antebraço para receber a classificação e o mapa de atenção do modelo.",
        eyebrow="Painel do Radiologista",
    )
    disclaimer(
        "<b>Ferramenta de apoio à decisão clínica.</b> Este resultado é uma estimativa automática e "
        "<b>não substitui</b> o laudo do médico radiologista responsável pela análise da anatomia do "
        "paciente e pela definição diagnóstica.",
        icon="🩺",
    )
    st.write("")

    if not model_is_available():
        st.error(
            "Nenhum modelo treinado foi encontrado em `models/`. "
            "Execute `python training/train_model.py` a partir da raiz do projeto "
            "antes de usar esta tela."
        )
        return

    _render_input()

    active = st.session_state.get("active_image")
    if active:
        _render_result(_analyze(active["name"], active["bytes"]))
    else:
        st.info("Envie uma radiografia ou escolha uma imagem de exemplo para começar.", icon=":material/info:")

    _render_history()
