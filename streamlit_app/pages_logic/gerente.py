"""Painel do Gerente/Chefe Clínico: métricas do modelo e visão de volume de
exames analisados pela equipe.

As análises espelham as seções de avaliação do notebook do projeto
(Laboratório/): matriz de confusão (19), Precision / Recall / F1 (20-22),
curva ROC (24), Precision-Recall e efeito do threshold (25), curvas de treino
(15-16), calibração (32) e limitações (35). Como no notebook, as métricas são
mostradas para a validação (estimativa mais realista) e para o teste
(provavelmente otimista, ver seção 10).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_metrics import load_curves, load_metrics
from theme import (
    CHART_AXIS,
    CHART_INK,
    CHART_MUTED,
    PLOTLY_CONFIG,
    SEQUENTIAL_BLUE_SCALE,
    SERIES_1,
    SERIES_2,
    base_chart_layout,
    disclaimer,
    kpi_grid,
    num,
    page_header,
    pct,
    section_title,
)

SPLIT_LABELS = {"train": "Treino", "validation": "Validação", "test": "Teste (15% do treino)"}
SET_OPTIONS = {"Validação": "validation", "Teste": "test"}
# Cores fixas por conjunto em todos os gráficos.
SET_COLORS = {"validation": SERIES_1, "test": SERIES_2}
GRID = "#e1e0d9"


def _chart(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def _set_metrics(metrics: dict, key: str) -> dict:
    """Métricas de um conjunto. As do teste ficam no nível de cima do JSON."""
    if key == "validation":
        return metrics["validation"]
    return {**metrics, "size": metrics["test_size"]}


# --------------------------------------------------------------- gráficos ----

def _confusion_matrix_figure(m: dict, labels) -> go.Figure:
    c = m["confusion_counts"]
    cm = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    names = [["Verdadeiro negativo", "Falso positivo"], ["Falso negativo", "Verdadeiro positivo"]]
    row_totals = [sum(row) for row in cm]
    z_max = max(max(row) for row in cm)

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=SEQUENTIAL_BLUE_SCALE,
            showscale=False,
            xgap=2,
            ygap=2,
            customdata=[[names[i][j] for j in range(2)] for i in range(2)],
            hovertemplate="%{customdata}<br>Real: %{y} · Previsto: %{x}<br>%{z} exames<extra></extra>",
        )
    )
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            fig.add_annotation(
                x=labels[j], y=labels[i], showarrow=False,
                text=(f"<b>{num(value)}</b><br><span style='font-size:11px'>{names[i][j].replace(' ', '<br>')}"
                      f"<br>{pct(value / row_totals[i])} da linha</span>"),
                font=dict(size=17, color="#ffffff" if value > z_max * 0.45 else CHART_INK),
            )
    fig.update_layout(**base_chart_layout(
        height=340,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(title="Previsto pelo modelo", side="bottom"),
        yaxis=dict(title="Classe real", autorange="reversed"),
    ))
    return fig


def _reference_line(x, y) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=CHART_AXIS, width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False,
    )


def _roc_figure(curves: dict) -> go.Figure:
    fig = go.Figure([_reference_line([0, 1], [0, 1])])
    for label, key in SET_OPTIONS.items():
        roc = curves[key]["roc"]
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{label} (AUC {num(roc['auc'], 3)})",
            line=dict(color=SET_COLORS[key], width=2),
            hovertemplate=f"{label}<br>Falsos positivos: %{{x:.1%}}<br>Sensibilidade: %{{y:.1%}}<extra></extra>",
        ))
    fig.add_annotation(x=0.62, y=0.5, text="chute aleatório", showarrow=False, textangle=-38,
                       font=dict(size=10, color=CHART_MUTED))
    fig.update_layout(**base_chart_layout(
        height=340,
        margin=dict(l=8, r=8, t=56, b=8),
        xaxis=dict(title="Taxa de falsos positivos", tickformat=".0%", range=[0, 1], gridcolor=GRID),
        yaxis=dict(title="Sensibilidade (recall)", tickformat=".0%", range=[0, 1.02], gridcolor=GRID),
        title=dict(text="Curva ROC", font=dict(size=13, color=CHART_INK), x=0.01, y=0.98),
        legend=dict(y=1.0),
    ))
    return fig


def _pr_figure(curves: dict) -> go.Figure:
    fig = go.Figure()
    for label, key in SET_OPTIONS.items():
        pr = curves[key]["pr"]
        fig.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines", name=f"{label} (AP {num(pr['average_precision'], 3)})",
            line=dict(color=SET_COLORS[key], width=2),
            hovertemplate=f"{label}<br>Recall: %{{x:.1%}}<br>Precisão: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(**base_chart_layout(
        height=340,
        margin=dict(l=8, r=8, t=56, b=8),
        xaxis=dict(title="Recall (sensibilidade)", tickformat=".0%", range=[0, 1], gridcolor=GRID),
        yaxis=dict(title="Precisão", tickformat=".0%", range=[0.4, 1.02], gridcolor=GRID),
        title=dict(text="Precision-Recall", font=dict(size=13, color=CHART_INK), x=0.01, y=0.98),
        legend=dict(y=1.0),
    ))
    return fig


def _calibration_figure(curves: dict) -> go.Figure:
    cal = curves["calibration"]
    fig = go.Figure([
        _reference_line([0, 1], [0, 1]),
        go.Scatter(
            x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers", name="Teste",
            line=dict(color=SET_COLORS["test"], width=2), marker=dict(size=8, line=dict(color="#ffffff", width=2)),
            showlegend=False,
            hovertemplate="Prob. média prevista: %{x:.0%}<br>Fração real com fratura: %{y:.0%}<extra></extra>",
        ),
    ])
    fig.add_annotation(x=0.3, y=0.36, text="calibração perfeita", showarrow=False, textangle=-38,
                       font=dict(size=10, color=CHART_MUTED))
    fig.update_layout(**base_chart_layout(
        height=330,
        xaxis=dict(title="Probabilidade prevista pelo modelo", tickformat=".0%", range=[0, 1], gridcolor=GRID),
        yaxis=dict(title="Fração real com fratura", tickformat=".0%", range=[0, 1.02], gridcolor=GRID),
        title=dict(text="Curva de calibração (teste)", font=dict(size=13, color=CHART_INK), x=0.01),
    ))
    return fig


def _history_figure(history: dict, metric: str, title: str, fmt: str, best_epoch: int) -> go.Figure:
    epochs = list(range(1, len(history[metric]) + 1))
    fig = go.Figure()
    for key, name, color in [(metric, "Treino", SERIES_1), (f"val_{metric}", "Validação", SERIES_2)]:
        fig.add_trace(go.Scatter(
            x=epochs, y=history[key], mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=8, line=dict(color="#ffffff", width=2)),
            hovertemplate=f"{name}: %{{y:{fmt}}}<extra></extra>",
        ))
    fig.add_vline(x=best_epoch, line=dict(color=CHART_AXIS, width=1.5, dash="dash"))
    fig.add_annotation(x=best_epoch, y=1, yref="paper", text="melhor época", showarrow=False,
                       xanchor="left", xshift=4, font=dict(size=10, color=CHART_MUTED))
    fig.update_layout(**base_chart_layout(
        height=300,
        hovermode="x unified",
        title=dict(text=title, font=dict(size=13, color=CHART_INK), x=0.01, y=0.98),
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(y=1.0, xanchor="right", x=1),
        xaxis=dict(title="Época", dtick=1, gridcolor=GRID),
        yaxis=dict(tickformat=fmt, gridcolor=GRID),
    ))
    return fig


def _distribution_figure(dataset_counts: dict) -> go.Figure:
    order = [s for s in ["train", "validation", "test"] if s in dataset_counts]
    x = [SPLIT_LABELS.get(s, s) for s in order]
    fig = go.Figure([
        go.Bar(name="Sem fratura", x=x, y=[dataset_counts[s]["not fractured"] for s in order],
               marker=dict(color=SERIES_1, line=dict(color="#fcfcfb", width=2)),
               hovertemplate="%{x}<br>Sem fratura: %{y:,}<extra></extra>"),
        go.Bar(name="Fraturado", x=x, y=[dataset_counts[s]["fractured"] for s in order],
               marker=dict(color=SERIES_2, line=dict(color="#fcfcfb", width=2)),
               hovertemplate="%{x}<br>Fraturado: %{y:,}<extra></extra>"),
    ])
    fig.update_layout(**base_chart_layout(
        height=320,
        barmode="group",
        bargap=0.35,
        bargroupgap=0.06,
        barcornerradius=4,
        separators=",.",
        yaxis=dict(title="Nº de imagens", gridcolor=GRID),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ------------------------------------------------------------------ abas ----

def _tab_desempenho(metrics: dict, m: dict, set_label: str):
    c = m["confusion_counts"]
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    labels = metrics.get("display_names", ["Sem fratura", "Fraturado"])
    col_cm, col_read = st.columns([1.2, 1], gap="large")
    with col_cm:
        section_title(f"Matriz de confusão — {set_label}",
                      "Linhas = classe real · colunas = previsão do modelo · % por linha.")
        _chart(_confusion_matrix_figure(m, labels))
    with col_read:
        section_title("Leitura dos erros")
        if fn > fp:
            balanco = (f"Com o limiar em {pct(metrics['threshold'], 0)}, o modelo erra mais por **omissão** "
                       "(fraturas não identificadas) do que por falso alarme.")
        elif fp > fn:
            balanco = (f"Com o limiar em {pct(metrics['threshold'], 0)}, o modelo erra mais por **falso alarme** "
                       "do que por omissão.")
        else:
            balanco = "Falsos positivos e falsos negativos ocorreram em quantidades iguais."
        st.markdown(
            f"- **{num(fn)} fraturas não identificadas** (falsos negativos) de {num(tp + fn)} "
            f"radiografias com fratura — {pct(fn / (tp + fn))} das fraturas reais.\n"
            f"- **{num(fp)} falsos alarmes** (falsos positivos) de {num(tn + fp)} radiografias "
            f"sem fratura — {pct(fp / (tn + fp))}.\n"
            f"- {balanco}"
        )
        precisao_nf = tn / (tn + fn) if (tn + fn) else 0
        especificidade = tn / (tn + fp) if (tn + fp) else 0
        f1_nf = (2 * precisao_nf * especificidade / (precisao_nf + especificidade)) if (precisao_nf + especificidade) else 0
        st.markdown("**Classe “Sem fratura”**")
        kpi_grid([
            {"label": "Precisão", "value": pct(precisao_nf)},
            {"label": "Especificidade", "value": pct(especificidade)},
            {"label": "F1-Score", "value": pct(f1_nf)},
        ])


def _tab_curvas(metrics: dict, curves):
    if curves is None or "validation" not in curves:
        st.info(
            "As curvas ainda não foram geradas. Execute `python training/evaluate_model.py` "
            "a partir da raiz do projeto.",
            icon=":material/info:",
        )
        return
    col_roc, col_pr = st.columns(2, gap="medium")
    with col_roc:
        _chart(_roc_figure(curves))
        st.caption("Quanto mais a curva se aproxima do canto superior esquerdo, melhor. "
                   "O AUC não depende do limiar.")
    with col_pr:
        _chart(_pr_figure(curves))
        st.caption("Mostra quanto de precisão se perde ao exigir mais sensibilidade.")

    col_cal, col_thr = st.columns(2, gap="medium")
    with col_cal:
        _chart(_calibration_figure(curves))
        st.caption(
            "Quanto mais perto da diagonal, mais as probabilidades coincidem com a frequência real de "
            "fraturas no teste. A probabilidade exibida ao radiologista não deve ser lida como risco clínico."
        )
    with col_thr:
        section_title(
            "Efeito do limiar de decisão (teste)",
            "Subir o limiar reduz falsos alarmes, mas deixa passar mais fraturas. "
            "Para escolher um limiar, use a validação, não o teste.",
        )
        df = pd.DataFrame(curves["threshold_table"])
        df.insert(0, "em uso", df["threshold"].round(2) == round(metrics["threshold"], 2))
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "em uso": st.column_config.CheckboxColumn("Atual", width="small"),
                "threshold": st.column_config.NumberColumn("Limiar", format="%.1f"),
                "precision": st.column_config.ProgressColumn("Precisão", min_value=0.0, max_value=1.0, format="percent"),
                "recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="percent"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "fn": st.column_config.NumberColumn("Fraturas perdidas"),
                "fp": st.column_config.NumberColumn("Falsos alarmes"),
            },
        )


def _tab_treinamento(metrics: dict):
    history = metrics.get("train_history")
    if not history:
        st.info("O histórico de treino deste modelo não está disponível (ele foi treinado no Colab). "
                "As curvas de acurácia e loss por época estão no notebook, seções 15 e 16.",
                icon=":material/info:")
        return
    best_epoch = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__) + 1
    col_acc, col_loss = st.columns(2, gap="medium")
    with col_acc:
        _chart(_history_figure(history, "accuracy", "Acurácia por época", ".0%", best_epoch))
    with col_loss:
        _chart(_history_figure(history, "loss", "Loss (binary cross-entropy)", ".2f", best_epoch))
    st.caption(
        f"{len(history['loss'])} épocas com Early Stopping (paciência 5, monitorando a loss de validação); "
        f"pesos restaurados da época {best_epoch}. Treino melhorando enquanto a validação estagna indica overfitting."
    )


def _tab_dataset(metrics: dict):
    counts = metrics["dataset_counts"]
    col_chart, col_info = st.columns([1.4, 1], gap="large")
    with col_chart:
        section_title("Imagens por conjunto e classe")
        _chart(_distribution_figure(counts))
    with col_info:
        section_title("Como os dados foram divididos")
        total = sum(v for split in counts.values() for v in split.values())
        st.markdown(
            f"- **{num(total)} radiografias** no total.\n"
            "- **Teste:** 15% de `train/`, sorteados imagem a imagem e estratificados por classe "
            "(semente 42).\n"
            "- **Validação:** pasta `validation/` original, de outra origem; usada no Early Stopping.\n"
            "- O dataset traz **cópias rotacionadas da mesma radiografia**: todas as imagens do teste "
            "têm uma \"irmã\" no treino, por isso o teste tende a ser otimista."
        )
        rows = [
            {"Conjunto": SPLIT_LABELS.get(s, s), "Sem fratura": v["not fractured"], "Fraturado": v["fractured"],
             "Total": v["not fractured"] + v["fractured"]}
            for s, v in counts.items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _tab_atividade():
    history = st.session_state.get("exam_history", [])
    st.caption(
        "Exames analisados nesta sessão do navegador pelo painel do(a) Radiologista. "
        "Não persiste entre reinicializações do aplicativo."
    )
    if not history:
        st.info("Nenhum exame foi analisado nesta sessão ainda.", icon=":material/hourglass_empty:")
        return
    df = pd.DataFrame(history)
    total = len(df)
    fraturados = int((df["resultado"] == "Fraturado").sum())
    kpi_grid([
        {"label": "Exames analisados", "value": num(total)},
        {"label": "Fraturas sinalizadas", "value": num(fraturados)},
        {"label": "Taxa de fratura", "value": pct(fraturados / total if total else 0)},
    ])
    st.dataframe(
        df.iloc[::-1],
        width="stretch",
        hide_index=True,
        column_config={
            "timestamp": "Data/hora", "arquivo": "Arquivo", "resultado": "Resultado", "confianca": "Confiança",
            "prob_fratura": st.column_config.ProgressColumn("Prob. fratura", min_value=0.0, max_value=1.0, format="percent"),
        },
    )


# ----------------------------------------------------------------- página ----

def render():
    page_header(
        "Dashboard Gerencial",
        "Desempenho da CNN na validação e no teste, e atividade da equipe.",
        eyebrow="Gerente / Chefe Clínico",
    )

    metrics = load_metrics()
    if metrics is None or "validation" not in metrics:
        st.error(
            "Métricas não encontradas em `models/metrics.json`. Execute "
            "`python training/evaluate_model.py` a partir da raiz do projeto."
        )
        return

    set_label = st.segmented_control(
        "Conjunto avaliado", list(SET_OPTIONS), default="Validação", key="gerente_conjunto",
    ) or "Validação"
    set_key = SET_OPTIONS[set_label]
    m = _set_metrics(metrics, set_key)

    if set_key == "validation":
        disclaimer(
            "<b>Validação: estimativa mais realista.</b> Imagens de outra origem, que a rede não usou para "
            "ajustar os pesos (servem só para o Early Stopping).",
            icon="✅",
        )
    else:
        disclaimer(
            "<b>Teste: provavelmente otimista.</b> Foi sorteado do próprio treino, imagem a imagem, e o dataset "
            "traz cópias rotacionadas da mesma radiografia; todas as imagens do teste têm uma \"irmã\" no treino.",
            icon="⚠️",
        )

    kpi_grid([
        {"label": "Acurácia", "value": pct(m["accuracy"]), "hint": "acertos nas duas classes", "highlight": True},
        {"label": "Precisão (Fratura)", "value": pct(m["precision"]),
         "hint": "alertas de fratura que estavam corretos"},
        {"label": "Recall / Sensibilidade", "value": pct(m["recall"]),
         "hint": "fraturas reais que o modelo detecta"},
        {"label": "F1-Score (Fratura)", "value": pct(m["f1_score"]),
         "hint": "equilíbrio precisão × recall"},
        {"label": "AUC (ROC)", "value": num(m["auc"], 3),
         "hint": "separação entre as classes (1 = perfeita)"},
    ])
    st.caption(
        f"{set_label}: {num(m['size'])} radiografias · {metrics.get('model_architecture', '')} · "
        f"{num(metrics.get('model_params', 0))} parâmetros · limiar {pct(metrics['threshold'], 0)}"
    )

    tabs = st.tabs([
        ":material/grid_view: Desempenho",
        ":material/show_chart: Curvas",
        ":material/model_training: Treinamento",
        ":material/dataset: Dataset",
        ":material/groups: Equipe",
    ])
    with tabs[0]:
        _tab_desempenho(metrics, m, set_label)
    with tabs[1]:
        _tab_curvas(metrics, load_curves())
    with tabs[2]:
        _tab_treinamento(metrics)
    with tabs[3]:
        _tab_dataset(metrics)
    with tabs[4]:
        _tab_atividade()

    st.write("")
    with st.expander("Limitações do modelo (leia antes de usar os números)", icon=":material/warning:"):
        st.markdown(
            "- **Cópias rotacionadas no teste:** o teste foi sorteado do treino imagem a imagem, e cópias "
            "rotacionadas da mesma radiografia ficaram nos dois conjuntos. Por isso a validação é a referência.\n"
            "- **Mudança de distribuição:** a validação vem de outra origem; as imagens *Sem fratura* são bem "
            "mais claras nela do que no treino.\n"
            "- **Overfitting:** a acurácia de treino sobe enquanto a de validação estagna.\n"
            "- **Probabilidades não calibradas:** ver curva de calibração na aba *Curvas*.\n"
            "- **Sem validação clínica:** não houve avaliação por especialistas nem teste em dados externos. "
            "Não é um dispositivo médico."
        )
