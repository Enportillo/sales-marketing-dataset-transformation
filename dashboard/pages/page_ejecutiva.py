"""Vista Ejecutiva – Resumen Estratégico con selector básico ES/EN."""

from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from dashboard.data_loader import (
    get_campaign_scores,
    get_classification_results,
    EXEC_AUC_READY,
    EXEC_AUC_MONITOR,
)
from dashboard.i18n import tr, normalize_lang
from dashboard.plot_helpers import add_non_overlapping_vline_labels

DEFAULT_THRESHOLD = 0.34


def _t(text: str, lang: str) -> str:
    """Alias local para traduccion centralizada."""
    return tr(text, lang)


def layout():
    return html.Div([
        html.Div([
            html.H1(id="exec-main-title", children="🏛️ Vista Ejecutiva"),
            html.P(
                id="exec-main-subtitle",
                children=(
                    "Resumen estratégico para toma de decisiones: cobertura, impacto potencial "
                    "y estado general del modelo de conversión."
                ),
            ),
        ], className="page-header"),

        html.Div([
            html.Span(id="exec-threshold-label", children="Escenario de umbral ejecutivo:", className="control-label"),
            dcc.Slider(
                id="exec-threshold",
                min=0.10,
                max=0.95,
                step=0.01,
                value=DEFAULT_THRESHOLD,
                marks={v: f"{int(v * 100)}%" for v in [0.10, 0.25, 0.34, 0.50, 0.75, 0.90]},
            ),
        ], className="dashboard-card"),

        html.Div(id="exec-kpis", className="kpi-row"),

        html.Div(id="exec-model-info", className="insight-box"),

        html.Div([
            html.H3(id="exec-eval-title", children="Resumen Ejecutivo de Evaluación"),
            html.P(
                id="exec-eval-desc",
                children="Lectura de desempeño del modelo para decisión estratégica (versión resumida).",
                className="card-desc",
            ),
            html.Div(id="exec-eval-summary", className="insight-box"),
        ], className="dashboard-card"),

        html.Div([
            html.H3(id="exec-dist-title", children="Distribución del Score de Conversión"),
            html.P(
                id="exec-dist-desc",
                children=(
                    "Segmentación global de clientes y corte de decisión para priorización "
                    "de inversión comercial."
                ),
                className="card-desc",
            ),
            dcc.Graph(id="exec-score-dist"),
        ], className="dashboard-card"),

        html.Div(className="grid-2", children=[
            html.Div([
                html.H3(id="exec-channel-title", children="Impacto por Canal"),
                dcc.Graph(id="exec-channel-impact"),
            ], className="dashboard-card"),
            html.Div([
                html.H3(id="exec-country-title", children="Impacto por País"),
                dcc.Graph(id="exec-country-impact"),
            ], className="dashboard-card"),
        ]),
    ], className="exec-view")


def register_callbacks(app):
    @app.callback(
        Output("exec-main-title", "children"),
        Output("exec-main-subtitle", "children"),
        Output("exec-threshold-label", "children"),
        Output("exec-eval-title", "children"),
        Output("exec-eval-desc", "children"),
        Output("exec-dist-title", "children"),
        Output("exec-dist-desc", "children"),
        Output("exec-channel-title", "children"),
        Output("exec-country-title", "children"),
        Output("exec-kpis", "children"),
        Output("exec-model-info", "children"),
        Output("exec-eval-summary", "children"),
        Output("exec-score-dist", "figure"),
        Output("exec-channel-impact", "figure"),
        Output("exec-country-impact", "figure"),
        Input("exec-threshold", "value"),
        Input("global-lang", "value"),
    )
    def update_exec(threshold, lang):
        lang = normalize_lang(lang)
        df = get_campaign_scores()
        if df.empty:
            empty = _empty_fig(_t("Datos no disponibles", lang))
            return (
                _t("🏛️ Vista Ejecutiva", lang),
                _t("Resumen estratégico para toma de decisiones: cobertura, impacto potencial y estado general del modelo de conversión.", lang),
                _t("Escenario de umbral ejecutivo:", lang),
                _t("Resumen Ejecutivo de Evaluación", lang),
                _t("Lectura de desempeño del modelo para decisión estratégica (versión resumida).", lang),
                _t("Distribución del Score de Conversión", lang),
                _t("Segmentación global de clientes y corte de decisión para priorización de inversión comercial.", lang),
                _t("Impacto por Canal", lang),
                _t("Impacto por País", lang),
                [],
                _t("Modelo no disponible", lang),
                _t("Resumen no disponible", lang),
                empty,
                empty,
                empty,
            )

        df = df.copy()
        thr = float(threshold if threshold is not None else DEFAULT_THRESHOLD)

        df["is_target"] = df["conversion_score"] >= thr
        df_target = df[df["is_target"]]

        n_total = len(df)
        n_target = len(df_target)
        cobertura = (n_target / n_total * 100.0) if n_total else 0.0
        score_medio = float(df["conversion_score"].mean()) if n_total else 0.0
        score_medio_target = float(df_target["conversion_score"].mean()) if n_target else 0.0
        spend_target = float(df_target["total_spent"].sum()) if (n_target and "total_spent" in df_target.columns) else 0.0

        cls_res = get_classification_results()
        model_name = type(cls_res.get("pipe_rf")).__name__ if cls_res.get("pipe_rf") is not None else _t("No disponible", lang)
        auc_rf = cls_res.get("auc_rf", None)
        auc_lr = cls_res.get("auc_lr", None)
        auc_text = f"AUC RF: {auc_rf:.4f}" if isinstance(auc_rf, float) else _t("AUC RF: N/D", lang)

        reg_results = cls_res.get("reg_results", {}) if isinstance(cls_res, dict) else {}
        rf_reg = reg_results.get("Random Forest Regressor", {})
        lr_reg = reg_results.get("Regresión Lineal", {})
        rf_r2 = rf_reg.get("R2", None)
        rf_rmse = rf_reg.get("RMSE", None)
        lr_r2 = lr_reg.get("R2", None)

        if isinstance(auc_rf, float) and auc_rf >= EXEC_AUC_READY:
            estado_modelo = _t("Adecuado para escalar campaña", lang)
        elif isinstance(auc_rf, float) and auc_rf >= EXEC_AUC_MONITOR:
            estado_modelo = _t("Apto con monitoreo de umbral y presupuesto", lang)
        else:
            estado_modelo = _t("Requiere mejora antes de escalar inversión", lang)

        eval_summary = html.Div([
            html.P([
                html.Strong(_t("Clasificación: ", lang)),
                (
                    f"AUC Random Forest = {auc_rf:.4f} | AUC Regresión Logística = {auc_lr:.4f}"
                    if isinstance(auc_rf, float) and isinstance(auc_lr, float)
                    else _t("Métricas AUC no disponibles", lang)
                ),
            ]),
            html.P([
                html.Strong(_t("Regresión (gasto): ", lang)),
                (
                    f"RF Regressor RMSE = {rf_rmse:.2f}, R² = {rf_r2:.4f} "
                    f"(benchmark lineal R² = {lr_r2:.4f})"
                    if isinstance(rf_rmse, float) and isinstance(rf_r2, float) and isinstance(lr_r2, float)
                    else _t("Métricas de regresión no disponibles", lang)
                ),
            ]),
            html.P([
                html.Strong(_t("Recomendación ejecutiva: ", lang)),
                estado_modelo,
            ]),
            html.P([
                html.Strong(_t("Umbrales vigentes (AUC RF): ", lang)),
                f"Escalar >= {EXEC_AUC_READY:.2f} | Monitoreo >= {EXEC_AUC_MONITOR:.2f}",
            ]),
            html.P(
                _t("Para análisis técnico detallado (matrices, curvas, tuning), consultar la Vista Técnica.", lang),
                style={"fontSize": "0.84rem", "color": "#555"},
            ),
            html.Div([
                html.A(
                    _t("Ver detalle técnico de evaluación →", lang),
                    href="/evaluacion",
                    style={
                        "display": "inline-block",
                        "marginTop": "8px",
                        "fontWeight": "700",
                        "fontSize": "0.88rem",
                        "color": "#2563eb",
                        "textDecoration": "none",
                    },
                )
            ]),
        ])

        kpis = [
            html.Div([
                html.Div(f"{n_total:,}", className="kpi-value"),
                html.Div(_t("Usuarios evaluados", lang), className="kpi-label"),
            ], className="kpi-card blue"),
            html.Div([
                html.Div(f"{n_target:,}", className="kpi-value"),
                html.Div(_t("Usuarios objetivo", lang), className="kpi-label"),
            ], className="kpi-card green"),
            html.Div([
                html.Div(f"{cobertura:.1f}%", className="kpi-value"),
                html.Div(_t("Cobertura objetivo", lang), className="kpi-label"),
            ], className="kpi-card orange"),
            html.Div([
                html.Div(f"{score_medio_target:.3f}", className="kpi-value"),
                html.Div(_t("Score medio objetivo", lang), className="kpi-label"),
            ], className="kpi-card purple"),
            html.Div([
                html.Div(f"${spend_target:,.0f}", className="kpi-value"),
                html.Div(_t("Gasto objetivo agregado", lang), className="kpi-label"),
            ], className="kpi-card red"),
        ]

        model_info = (
            f"{_t('Modelo estratégico vigente:', lang)} {model_name}. {auc_text}. "
            f"{_t('Score promedio global:', lang)} {score_medio:.3f}."
        )

        fig_dist = px.histogram(
            df,
            x="conversion_score",
            color="is_target",
            nbins=40,
            barmode="overlay",
            opacity=0.8,
            title=_t("Distribución de Score y Cobertura Objetivo", lang),
            labels={
                "conversion_score": _t("Score de conversión", lang),
                "is_target": _t("Objetivo", lang),
            },
            color_discrete_map={True: "#22c55e", False: "#94a3b8"},
        )
        add_non_overlapping_vline_labels(
            fig_dist,
            lines=[
                {
                    "x": thr,
                    "text": f"{_t('Umbral ejecutivo', lang)}: {thr:.2f}",
                    "line_color": "#ef4444",
                    "line_dash": "dash",
                }
            ],
            data_min=float(df["conversion_score"].min()),
            data_max=float(df["conversion_score"].max()),
        )
        fig_dist.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=80, b=40, l=60, r=20),
            height=440,
        )

        if "acquisition_channel" in df.columns:
            by_channel = df.groupby("acquisition_channel", as_index=False).agg(
                score_promedio=("conversion_score", "mean"),
                usuarios_objetivo=("is_target", "sum"),
            )
            by_channel = by_channel.sort_values("score_promedio", ascending=False)
            fig_channel = px.bar(
                by_channel,
                x="acquisition_channel",
                y="score_promedio",
                text="usuarios_objetivo",
                title=_t("Score Promedio por Canal", lang),
                color_discrete_sequence=["#3b82f6"],
            )
            fig_channel.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="#fafafa",
                font_family="Segoe UI",
                margin=dict(t=60, b=60, l=60, r=20),
                height=440,
                xaxis=dict(tickangle=-30),
            )
        else:
            fig_channel = _empty_fig(_t("Columna acquisition_channel no disponible", lang))

        if "country" in df.columns:
            by_country = df.groupby("country", as_index=False).agg(
                score_promedio=("conversion_score", "mean"),
                usuarios_objetivo=("is_target", "sum"),
            )
            by_country = by_country.sort_values("score_promedio", ascending=False).head(15)
            fig_country = px.bar(
                by_country,
                x="country",
                y="score_promedio",
                text="usuarios_objetivo",
                title=_t("Top Países por Score Promedio", lang),
                color_discrete_sequence=["#10b981"],
            )
            fig_country.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="#fafafa",
                font_family="Segoe UI",
                margin=dict(t=60, b=60, l=60, r=20),
                height=440,
                xaxis=dict(tickangle=-20),
            )
        else:
            fig_country = _empty_fig(_t("Columna country no disponible", lang))

        return (
            _t("🏛️ Vista Ejecutiva", lang),
            _t("Resumen estratégico para toma de decisiones: cobertura, impacto potencial y estado general del modelo de conversión.", lang),
            _t("Escenario de umbral ejecutivo:", lang),
            _t("Resumen Ejecutivo de Evaluación", lang),
            _t("Lectura de desempeño del modelo para decisión estratégica (versión resumida).", lang),
            _t("Distribución del Score de Conversión", lang),
            _t("Segmentación global de clientes y corte de decisión para priorización de inversión comercial.", lang),
            _t("Impacto por Canal", lang),
            _t("Impacto por País", lang),
            kpis,
            model_info,
            eval_summary,
            fig_dist,
            fig_channel,
            fig_country,
        )


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#999"),
    )
    fig.update_layout(paper_bgcolor="white")
    return fig
