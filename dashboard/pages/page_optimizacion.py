"""
Página 6 – Optimización de Hiperparámetros
"""

from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_loader import get_optimization_results
from dashboard.i18n import tr, normalize_lang
from dashboard.plot_helpers import add_hline_label

PALETTE = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]
VERSION_COLORS = {
    "v1 - Todas las features": "#3b82f6",
    "v2 - Solo comportamiento": "#22c55e",
    "v3 - Annual como positivo": "#f59e0b",
}


def layout(lang=None):
    lang = normalize_lang(lang)
    return html.Div([
        # ── Encabezado ────────────────────────────────────────────────────────
        html.Div([
            html.H1(tr("⚙️ Optimización de Hiperparámetros", lang)),
            html.P(tr("Comparativa de versiones del modelo Random Forest con "
                      "GridSearchCV. Evolución del ROC-AUC por versión.", lang)),
        ], className="page-header"),

        html.Div([
            tr(
                "⏳ Este módulo ejecuta GridSearchCV al cargar la página. "
                "El proceso puede tardar 1-2 minutos. "
                "Los resultados se almacenan en caché tras la primera ejecución.",
                lang,
            )
        ], className="insight-box warning"),

        # ── KPIs de versiones ─────────────────────────────────────────────────
        html.Div(id="opt-kpi-row", className="kpi-row"),

        # ── Evolución del AUC ─────────────────────────────────────────────────
        html.Div([
                 html.H3(tr("Evolución del ROC-AUC por Versión", lang)),
                 html.P(tr("Comparativa de rendimiento entre las versiones del modelo.", lang),
                   className="card-desc"),
            dcc.Graph(id="opt-auc-evolution"),
            html.Div([
                tr(
                    "💡 Cada versión explora una hipótesis distinta: "
                    "v1 usa todas las features, v2 solo comportamiento, "
                    "v3 redefine el target (Annual como positivo).",
                    lang,
                )
            ], className="insight-box"),
        ], className="dashboard-card"),

        # ── Matrices de confusión por versión ─────────────────────────────────
        html.Div([
            html.H3(tr("Matrices de Confusión por Versión", lang)),
            html.P(tr("Distribución de errores de cada variante del modelo.", lang),
                   className="card-desc"),
            html.Div([
                html.Span(tr("Seleccionar versión:", lang), className="control-label"),
                dcc.Dropdown(
                    id="opt-version-select",
                    options=[],   # se pobla en callback
                    value=None,
                    clearable=False,
                    style={"minWidth": "250px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="opt-cm-chart"),
        ], className="dashboard-card"),

        # ── Mejores hiperparámetros ────────────────────────────────────────────
        html.Div([
                 html.H3(tr("Mejores Hiperparámetros por Versión", lang)),
                 html.P(tr("Resultado de GridSearchCV (cv=3, scoring='roc_auc').", lang),
                   className="card-desc"),
            html.Div(id="opt-params-table"),
        ], className="dashboard-card"),

        # ── Comparativa de métricas ───────────────────────────────────────────
        html.Div([
                 html.H3(tr("Comparativa de Precision, Recall y F1 por Versión", lang)),
                 html.P(tr("Métricas de clasificación en el conjunto de test.", lang),
                   className="card-desc"),
            dcc.Graph(id="opt-metrics-chart"),
        ], className="dashboard-card"),

        # ── Resumen ejecutivo ─────────────────────────────────────────────────
        html.Div([
            html.H3(tr("Resumen y Decisión Final", lang)),
            html.Div(id="opt-summary-text"),
        ], className="dashboard-card"),
    ])


def register_callbacks(app):

    @app.callback(
        Output("opt-kpi-row", "children"),
        Output("opt-version-select", "options"),
        Output("opt-version-select", "value"),
        Output("opt-auc-evolution", "figure"),
        Output("opt-params-table", "children"),
        Output("opt-metrics-chart", "figure"),
        Output("opt-summary-text", "children"),
        Input("opt-version-select", "id"),   # trigger al cargar
        Input("global-lang", "value"),
    )
    def load_results(_=None, lang=None):
        lang = normalize_lang(lang)
        res = get_optimization_results()
        versions = res["versions"]

        # ── KPIs ──────────────────────────────────────────────────────────────
        kpis = []
        colors = ["blue", "green", "orange", "purple", "red"]
        best_idx = max(range(len(versions)), key=lambda i: versions[i]["auc"])
        for i, v in enumerate(versions):
            kpis.append(html.Div([
                html.Div(f"{v['auc']:.4f}", className="kpi-value"),
                html.Div(v["label"], className="kpi-label"),
            ], className=f"kpi-card {colors[i % len(colors)]}",
               style={"border": "2px solid #f59e0b" if i == best_idx else "none"}))

        # ── Opciones dropdown ─────────────────────────────────────────────────
        options = [{"label": v["label"], "value": i} for i, v in enumerate(versions)]
        value = 0

        # ── Gráfico de evolución AUC ──────────────────────────────────────────
        df_auc = pd.DataFrame({
            "Versión": [v["label"] for v in versions],
            "ROC-AUC": [v["auc"] for v in versions],
        })
        fig_auc = px.bar(
            df_auc, x="Versión", y="ROC-AUC",
            title=tr("ROC-AUC por Versión del Modelo", lang),
            color="Versión",
            color_discrete_sequence=PALETTE,
            text_auto=".4f",
        )
        add_hline_label(
            fig_auc,
            y=0.5,
            text=tr("Línea base aleatoria (0.50)", lang),
            line_color="#999999",
            line_dash="dash",
        )
        fig_auc.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI", showlegend=False,
            margin=dict(t=65, b=100, l=60, r=20),
            yaxis_range=[0.45, max(v["auc"] for v in versions) + 0.05],
        )

        # ── Tabla de hiperparámetros ──────────────────────────────────────────
        rows = []
        for v in versions:
            row = {"Versión": v["label"]}
            for k, val in v["best_params"].items():
                clean_k = k.replace("clf__", "")
                row[clean_k] = str(val)
            rows.append(row)
        df_params = pd.DataFrame(rows)
        param_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_params.columns],
            data=df_params.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px 12px", "fontSize": "0.82rem",
                        "fontFamily": "Segoe UI", "textAlign": "left"},
            style_header={"backgroundColor": "#1a1a2e", "color": "white",
                          "fontWeight": "700"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
            ],
        )

        # ── Comparativa de métricas ───────────────────────────────────────────
        metric_rows = []
        for v in versions:
            report = v["report"]
            for clase in ["0", "1"]:
                if clase in report:
                    r = report[clase]
                    metric_rows.append({
                        "Versión": v["label"],
                        "Clase": f"Clase {clase}",
                        "Precision": round(r["precision"], 3),
                        "Recall": round(r["recall"], 3),
                        "F1-Score": round(r["f1-score"], 3),
                    })
        df_metrics = pd.DataFrame(metric_rows)
        df_melt = df_metrics.melt(
            id_vars=["Versión", "Clase"],
            var_name="Métrica", value_name="Valor"
        )
        df_melt["Versión_Clase"] = df_melt["Versión"] + " | " + df_melt["Clase"]

        fig_metrics = px.bar(
            df_melt, x="Métrica", y="Valor",
            color="Versión_Clase", barmode="group",
            title=tr("Precision / Recall / F1 por Versión y Clase", lang),
            color_discrete_sequence=px.colors.qualitative.Set1,
            text_auto=".2f",
        )
        fig_metrics.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=40, l=60, r=20),
            yaxis_range=[0, 1.05],
        )

        # ── Resumen ejecutivo ─────────────────────────────────────────────────
        best_v = versions[best_idx]
        summary = html.Div([
            html.P([
                html.Strong(tr("Mejor versión: ", lang)),
                f"{best_v['label']} con ROC-AUC = {best_v['auc']:.4f}"
            ]),
            html.P([
                html.Strong(tr("Mejores parámetros: ", lang)),
                str(best_v["best_params"])
            ]),
            html.P(
                tr(
                    "El modelo Random Forest demuestra capacidad discriminatoria moderada "
                    "en todas las versiones. La reducción de features (v2) y la redefinición "
                    "del target (v3) exploran distintas hipótesis para mejorar el rendimiento. "
                    "Se recomienda continuar con ingeniería de features y evaluar "
                    "modelos ensamblados adicionales.",
                    lang,
                ),
                style={"fontSize": "0.87rem", "color": "#444"},
            ),
        ])

        return kpis, options, value, fig_auc, param_table, fig_metrics, summary

    @app.callback(
        Output("opt-cm-chart", "figure"),
        Input("opt-version-select", "value"),
        Input("global-lang", "value"),
    )
    def update_cm(version_idx, lang):
        lang = normalize_lang(lang)
        if version_idx is None:
            return go.Figure()
        res = get_optimization_results()
        v = res["versions"][version_idx]
        cm = v["cm"]
        labels = ["Básico (0)", "Anual (1)"]

        fig = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            title=f"Conf. Matrix – {v['label']}",
            labels=dict(x=tr("Predicción", lang), y=tr("Real", lang)),
            aspect="equal",
        )
        tn, fp, fn, tp = cm.ravel()
        fig.add_annotation(
            text=f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
            xref="paper", yref="paper",
            x=0.5, y=-0.22, showarrow=False,
            font=dict(size=12, color="#555"),
        )
        fig.update_layout(
            paper_bgcolor="white",
            font_family="Segoe UI",
            margin=dict(t=50, b=110, l=60, r=20),
            height=420,
            coloraxis_showscale=False,
        )
        return fig
