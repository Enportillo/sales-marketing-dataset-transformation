"""
Página 5 – Evaluación de Modelos Supervisados
"""

from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_loader import get_classification_results
from dashboard.i18n import tr, normalize_lang
from dashboard.plot_helpers import add_non_overlapping_vline_labels

PALETTE = px.colors.qualitative.Set2


def layout(lang=None):
    lang = normalize_lang(lang)
    return html.Div([
        # ── Encabezado ────────────────────────────────────────────────────────
        html.Div([
            html.H1(tr("📉 Evaluación de Modelos Supervisados", lang)),
            html.P(tr("Comparativa entre Random Forest y Regresión Logística para "
                      "clasificación (subscription_type) y regresión (total_spent).", lang)),
        ], className="page-header"),

        # ── KPIs ──────────────────────────────────────────────────────────────
        html.Div(id="eval-kpi-row", className="kpi-row"),

        # ── Sección 1: Matrices de Confusión ──────────────────────────────────
        html.Div([
            html.H3(tr("Matrices de Confusión", lang)),
            html.P(tr("Verdaderos / Falsos positivos y negativos de cada modelo "
                      "en el conjunto de test.", lang), className="card-desc"),
            html.Div(className="grid-2", children=[
                html.Div([
                    html.Div(tr("🌲 Random Forest", lang), className="badge-rf",
                             style={"marginBottom": "10px", "display": "inline-block"}),
                    dcc.Graph(id="eval-cm-rf"),
                ]),
                html.Div([
                    html.Div(tr("📐 Regresión Logística", lang), className="badge-lr",
                             style={"marginBottom": "10px", "display": "inline-block"}),
                    dcc.Graph(id="eval-cm-lr"),
                ]),
            ]),
        ], className="dashboard-card"),

        # ── Sección 2: Curvas ROC ─────────────────────────────────────────────
        html.Div([
                 html.H3(tr("Curvas ROC", lang)),
                 html.P(tr("Relación TPR / FPR a distintos umbrales de decisión.", lang),
                   className="card-desc"),
            dcc.Graph(id="eval-roc-curves"),
            html.Div([
                "💡 Random Forest obtiene ROC-AUC ≈ 0.63 frente a ≈ 0.51 de Regresión Logística. "
                "El diferencial confirma que el problema requiere modelar relaciones no lineales."
            ], className="insight-box"),
        ], className="dashboard-card"),

        # ── Sección 3: Comparativa de métricas ───────────────────────────────
        html.Div([
                 html.H3(tr("Comparativa de Métricas de Clasificación", lang)),
                 html.P(tr("Precision, Recall y F1-Score por clase y modelo.", lang),
                   className="card-desc"),
            dcc.Graph(id="eval-metrics-compare"),
        ], className="dashboard-card"),

        # ── Sección 4: Distribución de probabilidades ─────────────────────────
        html.Div([
            html.H3(tr("Distribución de Probabilidades Predichas", lang)),
            html.P(tr("Histograma de las probabilidades asignadas a cada clase "
                      "para detectar calibración del modelo.", lang), className="card-desc"),
            html.Div([
                html.Span(tr("Modelo:", lang), className="control-label"),
                dcc.RadioItems(
                    id="eval-prob-model",
                    options=[
                        {"label": "  Random Forest", "value": "rf"},
                        {"label": "  Regresión Logística", "value": "lr"},
                    ],
                    value="rf",
                    inline=True,
                    style={"fontSize": "0.88rem"},
                ),
            ], className="control-row"),
            dcc.Graph(id="eval-prob-dist"),
        ], className="dashboard-card"),

        # ── Sección 5: Regresión ─────────────────────────────────────────────
        html.Div([
                 html.H3(tr("Modelos de Regresión – Predicción de total_spent", lang)),
                 html.P(tr("RMSE y R² de Regresión Lineal vs Random Forest Regressor.", lang),
                   className="card-desc"),
            html.Div(className="grid-2", children=[
                dcc.Graph(id="eval-reg-rmse"),
                dcc.Graph(id="eval-reg-r2"),
            ]),
            html.Div([
                "📊 RandomForestRegressor supera a Regresión Lineal en ambas métricas. "
                "Sin embargo, el R² bajo (~0.05) indica que el modelo está subexplicado "
                "y depende de enriquecer el espacio de variables."
            ], className="insight-box warning"),
        ], className="dashboard-card"),

        # ── Sección 6: Upselling ─────────────────────────────────────────────
        html.Div([
            html.H3(tr("Simulación de Campaña de Upselling", lang)),
            html.P(tr("Distribución de probabilidades de upgrade para clientes "
                      "actualmente en plan Básico.", lang), className="card-desc"),
            html.Div([
                html.Span(tr("Umbral de selección:", lang), className="control-label"),
                dcc.Slider(
                    id="eval-upsell-threshold",
                    min=0.1, max=0.95, step=0.05, value=0.75,
                    marks={v: f"{int(v*100)}%" for v in [0.1, 0.25, 0.5, 0.75, 0.9]},
                ),
            ], className="control-row"),
            dcc.Graph(id="eval-upsell-chart"),
            html.Div(id="eval-upsell-summary", className="insight-box success"),
        ], className="dashboard-card"),
    ])


def register_callbacks(app):

    @app.callback(
        Output("eval-kpi-row", "children"),
        Input("eval-prob-model", "value"),
        Input("global-lang", "value"),
    )
    def update_kpis(model, lang):
        lang = normalize_lang(lang)
        res = get_classification_results()
        auc_rf = res["auc_rf"]
        auc_lr = res["auc_lr"]
        reg = res["reg_results"]
        rf_reg = reg.get("Random Forest Regressor", {})
        lr_reg = reg.get("Regresión Lineal", {})

        return [
            html.Div([html.Div(f"{auc_rf:.4f}", className="kpi-value"),
                      html.Div(tr("AUC – Random Forest", lang), className="kpi-label")],
                     className="kpi-card blue"),
            html.Div([html.Div(f"{auc_lr:.4f}", className="kpi-value"),
                      html.Div(tr("AUC – Regresión Logística", lang), className="kpi-label")],
                     className="kpi-card purple"),
            html.Div([html.Div(f"{rf_reg.get('RMSE', 0):.1f}", className="kpi-value"),
                      html.Div(tr("RMSE – RF Regressor", lang), className="kpi-label")],
                     className="kpi-card orange"),
            html.Div([html.Div(f"{rf_reg.get('R2', 0):.4f}", className="kpi-value"),
                      html.Div(tr("R² – RF Regressor", lang), className="kpi-label")],
                     className="kpi-card green"),
        ]

    @app.callback(
        Output("eval-cm-rf", "figure"),
        Output("eval-cm-lr", "figure"),
        Input("eval-prob-model", "value"),
        Input("global-lang", "value"),
    )
    def update_cms(model, lang):
        lang = normalize_lang(lang)
        res = get_classification_results()
        return (
            _cm_fig(res["cm_rf"], tr("Conf. Matrix – Random Forest", lang), "Blues", lang),
            _cm_fig(res["cm_lr"], tr("Conf. Matrix – Reg. Logística", lang), "Purples", lang),
        )

    @app.callback(
        Output("eval-roc-curves", "figure"),
        Input("eval-prob-model", "value"),
        Input("global-lang", "value"),
    )
    def update_roc(model, lang):
        lang = normalize_lang(lang)
        from sklearn.metrics import roc_curve
        res = get_classification_results()
        y_test = res["y_test"]

        fig = go.Figure()
        for name, probs, color in [
            ("Random Forest", res["probs_rf"], "#3b82f6"),
            ("Reg. Logística", res["probs_lr"], "#8b5cf6"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_label = res["auc_rf"] if "Forest" in name else res["auc_lr"]
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC = {auc_label:.4f})",
                line=dict(color=color, width=2.5),
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random (AUC = 0.5)",
            line=dict(color="#999", width=1.5, dash="dash"),
        ))
        fig.update_layout(
            title=tr("Curvas ROC – Clasificación de subscription_type", lang),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=40, l=60, r=20),
            height=400,
        )
        return fig

    @app.callback(
        Output("eval-metrics-compare", "figure"),
        Input("eval-prob-model", "value"),
        Input("global-lang", "value"),
    )
    def update_metrics_compare(model, lang):
        lang = normalize_lang(lang)
        res = get_classification_results()
        rows = []
        for name, report in [("Random Forest", res["report_rf"]),
                               ("Reg. Logística", res["report_lr"])]:
            for clase in ["0", "1"]:
                if clase in report:
                    r = report[clase]
                    rows.append({
                        "Modelo": name,
                        "Clase": f"Clase {clase}",
                        "Precision": r["precision"],
                        "Recall": r["recall"],
                        "F1-Score": r["f1-score"],
                    })

        df_m = pd.DataFrame(rows)
        df_melt = df_m.melt(
            id_vars=["Modelo", "Clase"],
            var_name="Métrica", value_name="Valor"
        )
        df_melt["Modelo_Clase"] = df_melt["Modelo"] + " | " + df_melt["Clase"]

        fig = px.bar(
            df_melt, x="Métrica", y="Valor",
            color="Modelo_Clase",
            barmode="group",
            title=tr("Precision / Recall / F1-Score por Modelo y Clase", lang),
            color_discrete_sequence=px.colors.qualitative.Set1,
            text_auto=".2f",
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=40, l=60, r=20),
            yaxis_range=[0, 1.05],
        )
        return fig

    @app.callback(
        Output("eval-prob-dist", "figure"),
        Input("eval-prob-model", "value"),
        Input("global-lang", "value"),
    )
    def update_prob_dist(model, lang):
        lang = normalize_lang(lang)
        res = get_classification_results()
        probs = res["probs_rf"] if model == "rf" else res["probs_lr"]
        y_test = res["y_test"]
        label = "Random Forest" if model == "rf" else "Regresión Logística"

        df_prob = pd.DataFrame({"Probabilidad": probs, "Clase Real": y_test.values})
        df_prob["Clase Real"] = df_prob["Clase Real"].map({0: "Básico (0)", 1: "Anual (1)"})

        fig = px.histogram(
            df_prob, x="Probabilidad", color="Clase Real",
            nbins=40, barmode="overlay",
            title=f"{tr('Distribución de Probabilidades', lang)} – {label}",
            color_discrete_map={"Básico (0)": "#3b82f6", "Anual (1)": "#22c55e"},
            opacity=0.75,
        )
        add_non_overlapping_vline_labels(
            fig,
            lines=[
                {
                    "x": 0.5,
                    "text": tr("Umbral 0.5", lang),
                    "line_color": "#ef4444",
                    "line_dash": "dash",
                }
            ],
            data_min=0.0,
            data_max=1.0,
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=80, b=40, l=60, r=20),
        )
        return fig

    @app.callback(
        Output("eval-reg-rmse", "figure"),
        Output("eval-reg-r2", "figure"),
        Input("eval-prob-model", "value"),
        Input("global-lang", "value"),
    )
    def update_reg(model, lang):
        lang = normalize_lang(lang)
        res = get_classification_results()
        reg = res["reg_results"]
        df_reg = pd.DataFrame([
            {"Modelo": k, "RMSE": v["RMSE"], "R²": v["R2"]}
            for k, v in reg.items()
        ])

        fig_rmse = px.bar(
            df_reg, x="Modelo", y="RMSE",
            title=tr("RMSE por Modelo (menor = mejor)", lang),
            color="Modelo", color_discrete_sequence=PALETTE,
            text_auto=".2f",
        )
        fig_rmse.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            showlegend=False, font_family="Segoe UI",
            margin=dict(t=50, b=60),
        )

        fig_r2 = px.bar(
            df_reg, x="Modelo", y="R²",
            title=tr("R² por Modelo (mayor = mejor)", lang),
            color="Modelo", color_discrete_sequence=PALETTE,
            text_auto=".4f",
        )
        fig_r2.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            showlegend=False, font_family="Segoe UI",
            margin=dict(t=50, b=60),
        )
        return fig_rmse, fig_r2

    @app.callback(
        Output("eval-upsell-chart", "figure"),
        Output("eval-upsell-summary", "children"),
        Input("eval-upsell-threshold", "value"),
        Input("global-lang", "value"),
    )
    def update_upsell(threshold, lang):
        lang = normalize_lang(lang)
        res = get_classification_results()
        probs = res["probs_upsell"]
        n_basicos = res["n_basicos"]

        n_objetivo = int((probs >= threshold).sum())

        fig = px.histogram(
            probs, nbins=40,
            title=tr("Probabilidades de Upgrade – Clientes Básicos", lang),
            labels={"value": "P(upgrade)", "count": tr("Cantidad de clientes", lang)},
            color_discrete_sequence=["#3b82f6"],
            opacity=0.8,
        )
        add_non_overlapping_vline_labels(
            fig,
            lines=[
                {
                    "x": float(threshold),
                    "text": f"{tr('Umbral', lang)}: {threshold:.0%}",
                    "line_color": "#ef4444",
                    "line_dash": "dash",
                }
            ],
            data_min=float(np.min(probs)),
            data_max=float(np.max(probs)),
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=80, b=40, l=60, r=20),
        )

        summary = (
            f"{tr('Con umbral', lang)} {threshold:.0%}: {tr('se identifican', lang)} {n_objetivo:,} {tr('de', lang)} {n_basicos:,} "
            f"{tr('clientes básicos', lang)} ({n_objetivo/max(n_basicos,1)*100:.1f}%) "
            f"{tr('como candidatos de alta probabilidad de upgrade a plan Anual.', lang)}"
        )
        return fig, summary


def _cm_fig(cm, title, colorscale, lang):
    labels = ["Básico (0)", "Anual (1)"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale=colorscale,
        title=title,
        labels=dict(x=tr("Predicción", lang), y=tr("Real", lang), color=tr("Cantidad", lang)),
        aspect="equal",
    )
    fig.update_layout(
        paper_bgcolor="white",
        font_family="Segoe UI",
        margin=dict(t=50, b=40, l=60, r=20),
        height=320,
        coloraxis_showscale=False,
    )
    return fig
