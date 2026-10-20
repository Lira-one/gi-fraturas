"""Painel do Gerente/Chefe Clínico: métricas do modelo e visão de volume de
exames analisados pela equipe.

As análises espelham as seções de avaliação do notebook de referência
(Laboratório/Deteccao_Fraturas.ipynb): matriz de confusão (19), Precision /
Recall / F1 (20-22), curva ROC (24), Precision-Recall e efeito do threshold
(25), curvas de treino (15-16), calibração (32) e limitações (35).
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

SPLIT_LABELS = {"train": "Treino", "validation": "Validação", "test": "Teste (holdout)"}


def _chart(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


# --------------------------------------------------------------- gráficos ----

def _confusion_matrix_figure(metrics: dict) -> go.Figure:
    cm = metrics["confusion_matrix"]
    # Ordem = class_names (índice 0 = not fractured, índice 1 = fractured)
    labels = metrics.get("display_names", ["Sem fratura", "Fraturado"])
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
                text=(f"<b>{num(value)}</b><br><span style='font-size:11px'>{names[i][j]}"
                      f"<br>{pct(value / row_totals[i])} da linha</span>"),
                font=dict(size=17, color="#ffffff" if value > z_max * 0.45 else CHART_INK),
            )
    fig.update_layout(**base_chart_layout(
        height=340,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(title="Previsto pelo modelo", side="bottom", tickfont=dict(color=CHART_MUTED)),
        yaxis=dict(title="Classe real", autorange="reversed", tickfont=dict(color=CHART_MUTED)),
    ))
    return fig


def _reference_line(x, y, name: str) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=dict(color=CHART_AXIS, width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False,
    )


def _roc_figure(curves: dict, metrics: dict) -> go.Figure:
    roc = curves["roc"]
    c = metrics["confusion_counts"]
    op_fpr = c["fp"] / (c["fp"] + c["tn"])
    op_tpr = c["tp"] / (c["tp"] + c["fn"])
    fig = go.Figure([
        _reference_line([0, 1], [0, 1], "Chute aleatório"),
        go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines", name="MobileNetV2",
            line=dict(color=SERIES_1, width=2), showlegend=False,
            hovertemplate="Falsos positivos: %{x:.1%}<br>Sensibilidade: %{y:.1%}<extra></extra>",
        ),
        go.Scatter(
            x=[op_fpr], y=[op_tpr], mode="markers", name="Limiar atual",
            marker=dict(size=10, color=SERIES_1, line=dict(color="#ffffff", width=2)), showlegend=False,
            hovertemplate=f"Limiar {pct(metrics['threshold'], 0)}<br>FPR %{{x:.1%}} · TPR %{{y:.1%}}<extra></extra>",
        ),
    ])
    fig.add_annotation(x=op_fpr, y=op_tpr, text=f"limiar {pct(metrics['threshold'], 0)}", showarrow=False,
                       xanchor="left", xshift=9, yshift=-12, font=dict(size=11, color=CHART_INK))
    fig.add_annotation(x=0.62, y=0.5, text="chute aleatório", showarrow=False, textangle=-38,
                       font=dict(size=10, color=CHART_MUTED))
    fig.update_layout(**base_chart_layout(
        height=330,
        xaxis=dict(title="Taxa de falsos positivos", tickformat=".0%", range=[0, 1], gridcolor="#e1e0d9"),
        yaxis=dict(title="Sensibilidade (recall)", tickformat=".0%", range=[0, 1.02], gridcolor="#e1e0d9"),
        title=dict(text=f"Curva ROC · AUC {num(roc['auc'], 3)}", font=dict(size=13, color=CHART_INK), x=0.01),
    ))
    return fig


def _pr_figure(curves: dict) -> go.Figure:
    pr = curves["pr"]
    fig = go.Figure([
        _reference_line([0, 1], [pr["baseline"], pr["baseline"]], "Prevalência"),
        go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines", name="MobileNetV2",
            line=dict(color=SERIES_1, width=2), showlegend=False,
            hovertemplate="Recall: %{x:.1%}<br>Precisão: %{y:.1%}<extra></extra>",
        ),
    ])
    fig.add_annotation(x=0.02, y=pr["baseline"], text=f"prevalência de fraturas ({pct(pr['baseline'])})",
                       showarrow=False, xanchor="left", yshift=-10, font=dict(size=10, color=CHART_MUTED))
    fig.update_layout(**base_chart_layout(
        height=330,
        xaxis=dict(title="Recall (sensibilidade)", tickformat=".0%", range=[0, 1], gridcolor="#e1e0d9"),
        yaxis=dict(title="Precisão", tickformat=".0%", range=[0.4, 1.02], gridcolor="#e1e0d9"),
        title=dict(text=f"Precision-Recall · AP {num(pr['average_precision'], 3)}", font=dict(size=13, color=CHART_INK), x=0.01),
    ))
    return fig


def _calibration_figure(curves: dict) -> go.Figure:
    cal = curves["calibration"]
    fig = go.Figure([
        _reference_line([0, 1], [0, 1], "Calibração perfeita"),
        go.Scatter(
            x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers", name="MobileNetV2",
            line=dict(color=SERIES_1, width=2), marker=dict(size=8, line=dict(color="#ffffff", width=2)),
            showlegend=False,
            hovertemplate="Prob. média prevista: %{x:.0%}<br>Fração real com fratura: %{y:.0%}<extra></extra>",
        ),
    ])
    fig.add_annotation(x=0.3, y=0.36, text="calibração perfeita", showarrow=False, textangle=-38,
                       font=dict(size=10, color=CHART_MUTED))
    fig.update_layout(**base_chart_layout(
        height=330,
        xaxis=dict(title="Probabilidade prevista pelo modelo", tickformat=".0%", range=[0, 1], gridcolor="#e1e0d9"),
        yaxis=dict(title="Fração real com fratura", tickformat=".0%", range=[0, 1.02], gridcolor="#e1e0d9"),
        title=dict(text="Curva de calibração", font=dict(size=13, color=CHART_INK), x=0.01),
    ))
    return fig


def _history_figure(history: dict, metric: str, title: str, fmt: str) -> go.Figure:
    epochs = list(range(1, len(history[metric]) + 1))
    fig = go.Figure()
    for key, name, color in [(metric, "Treino", SERIES_1), (f"val_{metric}", "Validação", SERIES_2)]:
        fig.add_trace(go.Scatter(
            x=epochs, y=history[key], mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=8, line=dict(color="#ffffff", width=2)),
            hovertemplate=f"{name}: %{{y:{fmt}}}<extra></extra>",
        ))
    fig.update_layout(**base_chart_layout(
        height=300,
        hovermode="x unified",
        title=dict(text=title, font=dict(size=13, color=CHART_INK), x=0.01, y=0.98),
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis=dict(title="Época", dtick=1, gridcolor="#e1e0d9"),
        yaxis=dict(tickformat=fmt, gridcolor="#e1e0d9"),
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
        yaxis=dict(title="Nº de imagens", gridcolor="#e1e0d9"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ------------------------------------------------------------------ abas ----

def _tab_desempenho(metrics: dict):
    c = metrics["confusion_counts"]
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    col_cm, col_read = st.columns([1.2, 1], gap="large")
    with col_cm:
        section_title("Matriz de confusão", "Linhas = classe real · colunas = previsão do modelo · % por linha.")
        _chart(_confusion_matrix_figure(metrics))
    with col_read:
        section_title("Leitura clínica dos erros")
        st.markdown(
            f"- **{num(fn)} fraturas não identificadas** (falsos negativos) de {num(tp + fn)} "
            f"radiografias com fratura — {pct(fn / (tp + fn))} das fraturas reais.\n"
            f"- **{num(fp)} falsos alarmes** (falsos positivos) de {num(tn + fp)} radiografias "
            f"sem fratura — {pct(fp / (tn + fp))}.\n"
            f"- Com o limiar em {pct(metrics['threshold'], 0)}, o modelo privilegia a **sensibilidade**: "
            "erra mais por excesso (alarme) do que por omissão, o que é desejável numa ferramenta de triagem."
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
    if curves is None:
        st.info(
            "As curvas ainda não foram geradas. Execute `python training/evaluate_curves.py` "
            "a partir da raiz do projeto.",
            icon=":material/info:",
        )
        return
    col_roc, col_pr = st.columns(2, gap="medium")
    with col_roc:
        _chart(_roc_figure(curves, metrics))
        st.caption("Quanto mais a curva se aproxima do canto superior esquerdo, melhor. O ponto marca o limiar em uso.")
    with col_pr:
        _chart(_pr_figure(curves))
        st.caption("Mostra quanto de precisão se perde ao exigir mais sensibilidade. Útil quando os falsos alarmes têm custo.")

    col_cal, col_thr = st.columns(2, gap="medium")
    with col_cal:
        _chart(_calibration_figure(curves))
        st.caption(
            "Pontos abaixo da diagonal = o modelo **superestima** a chance de fratura naquela faixa. "
            "Por isso a probabilidade exibida ao radiologista não deve ser lida como risco clínico."
        )
    with col_thr:
        section_title(
            "Efeito do limiar de decisão",
            "Subir o limiar reduz falsos alarmes, mas deixa passar mais fraturas.",
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
        st.info("Histórico de treino indisponível em `metrics.json`.")
        return
    col_acc, col_loss = st.columns(2, gap="medium")
    with col_acc:
        _chart(_history_figure(history, "accuracy", "Acurácia por época", ".0%"))
    with col_loss:
        _chart(_history_figure(history, "loss", "Loss (binary cross-entropy)", ".2f"))
    st.caption(
        f"{len(history['loss'])} épocas com Early Stopping (paciência 5, monitorando a loss de validação). "
        "Uma distância grande e crescente entre treino e validação indicaria overfitting."
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
            "- **Teste (holdout):** 15% de `train/`, estratificado por classe, nunca visto no treino.\n"
            "- **Validação:** pasta `validation/` original, usada só para o Early Stopping.\n"
            "- Classes equilibradas no treino, então a acurácia não é inflada por desbalanceamento."
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
        "Desempenho do modelo no conjunto de teste e atividade da equipe.",
        eyebrow="Gerente / Chefe Clínico",
    )

    metrics = load_metrics()
    if metrics is None:
        st.error(
            "Nenhuma métrica encontrada em `models/metrics.json`. "
            "Execute `python training/train_model.py` a partir da raiz do projeto "
            "para treinar o modelo e gerar os resultados."
        )
        return

    kpi_grid([
        {"label": "Recall / Sensibilidade", "value": pct(metrics["recall"]),
         "hint": "fraturas reais que o modelo detecta", "highlight": True},
        {"label": "Precisão (Fratura)", "value": pct(metrics["precision"]),
         "hint": "alertas de fratura que estavam corretos"},
        {"label": "F1-Score (Fratura)", "value": pct(metrics["f1_score"]),
         "hint": "equilíbrio precisão × recall"},
        {"label": "Acurácia geral", "value": pct(metrics["accuracy"]),
         "hint": "acertos nas duas classes"},
        {"label": "AUC (ROC)", "value": num(metrics.get("auc", 0), 3),
         "hint": "separação entre as classes (1 = perfeita)"},
    ])
    st.caption(
        f"Avaliado em {num(metrics['test_size'])} radiografias de teste (holdout real) · "
        f"{metrics.get('model_architecture', '')} · treinado em "
        f"{pd.Timestamp(metrics['trained_at']).strftime('%d/%m/%Y')}"
    )

    tabs = st.tabs([
        ":material/grid_view: Desempenho",
        ":material/show_chart: Curvas",
        ":material/model_training: Treinamento",
        ":material/dataset: Dataset",
        ":material/groups: Equipe",
    ])
    with tabs[0]:
        _tab_desempenho(metrics)
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
            "- **Mesmo dataset:** o desempenho vale para imagens parecidas com as do treino; "
            "radiografias de outros equipamentos ou projeções podem ter desempenho bem pior.\n"
            "- **Possível viés de fonte:** se cada classe veio de origens diferentes, a rede pode aprender "
            "características da fonte (contraste, marcações, gesso) em vez da fratura.\n"
            "- **Probabilidades não calibradas:** ver curva de calibração na aba *Curvas*.\n"
            "- **Sem validação clínica:** não houve avaliação por especialistas, análise de subgrupos "
            "nem estudo prospectivo. Não é um dispositivo médico."
        )
    disclaimer(
        "Métricas calculadas para a classe <b>Fraturado</b> (classe de interesse clínico) no conjunto de "
        "teste, que nunca foi usado no treino. Fonte: <code>models/metrics.json</code> e "
        "<code>models/evaluation_curves.json</code>.",
        icon="📌",
    )
