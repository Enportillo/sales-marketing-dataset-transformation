"""
Vista Operativa – Ejecución de Campañas
"""

from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from dash.dash_table.Format import Format, Group, Scheme, Symbol

from dashboard.data_loader import get_campaign_scores, get_classification_results
from dashboard.i18n import tr, normalize_lang
from dashboard.plot_helpers import add_non_overlapping_vline_labels

DEFAULT_THRESHOLD = 0.34

FRIENDLY_COL_NAMES = {
    "client_id": "ID Cliente",
    "score_band": "Semáforo",
    "conversion_score": "Prob. Conversión",
    "score_percentil": "Percentil Score",
    "gender": "Género",
    "subscription_type": "Tipo Suscripción",
    "payment_method": "Método de Pago",
    "total_spent": "Gasto Total",
    "avg_order_value": "Ticket Promedio",
    "last_3_month_purchase_freq": "Freq. Compra (3M)",
    "total_visits": "Visitas Totales",
    "pages_per_session": "Páginas/Sesión",
    "support_tickets": "Tickets Soporte",
    "acquisition_channel": "Canal Adquisición",
    "country": "País",
}

COL_FORMATTERS = {
    "conversion_score": Format(precision=2, scheme=Scheme.percentage),
    "score_percentil": Format(precision=1, scheme=Scheme.fixed),
    "total_spent": Format(
        precision=0,
        scheme=Scheme.fixed,
        group=Group.yes,
        groups=3,
        symbol=Symbol.yes,
        symbol_prefix="$",
    ),
    "avg_order_value": Format(
        precision=2,
        scheme=Scheme.fixed,
        group=Group.yes,
        groups=3,
        symbol=Symbol.yes,
        symbol_prefix="$",
    ),
}

COL_TYPES = {
    "conversion_score": "numeric",
    "score_percentil": "numeric",
    "total_spent": "numeric",
    "avg_order_value": "numeric",
    "last_3_month_purchase_freq": "numeric",
    "total_visits": "numeric",
    "pages_per_session": "numeric",
    "support_tickets": "numeric",
}


def layout():
    return html.Div([
        html.Div([
            html.H1(id="ops-main-title", children="🛠️ Vista Operativa"),
            html.P(
                id="ops-main-subtitle",
                children=(
                    "Ejecución diaria de campañas: priorización de clientes, monitoreo del "
                    "flujo operativo y segmentación accionable."
                ),
            ),
        ], className="page-header"),

        html.Div([
            html.Div([
                html.Span(id="ops-threshold-label", children="Umbral operativo:", className="control-label"),
                dcc.Slider(
                    id="ops-threshold",
                    min=0.10,
                    max=0.95,
                    step=0.01,
                    value=DEFAULT_THRESHOLD,
                    marks={v: f"{int(v * 100)}%" for v in [0.10, 0.25, 0.34, 0.50, 0.75, 0.90]},
                ),
            ], style={"flex": "2"}),
            html.Div([
                html.Span(id="ops-topn-label", children="Top usuarios a mostrar:", className="control-label"),
                dcc.Dropdown(
                    id="ops-topn",
                    options=[{"label": str(n), "value": n} for n in [10, 25, 50, 100, 250]],
                    value=50,
                    clearable=False,
                    style={"minWidth": "180px"},
                ),
            ], style={"flex": "1", "minWidth": "220px"}),
        ], className="dashboard-card control-row"),

        html.Div(id="ops-kpis", className="kpi-row"),

        html.Div(id="ops-model-info", className="insight-box"),

        html.Div([
            html.H3(id="ops-dist-title", children="Distribución del Score de Conversión"),
            html.P(
                id="ops-dist-desc",
                children=(
                    "Visualiza el corte dinámico del umbral y cuántos usuarios quedan "
                    "como objetivo de campaña."
                ),
                className="card-desc",
            ),
            dcc.Graph(id="ops-score-dist"),
        ], className="dashboard-card"),

        html.Div([
            html.H3(id="ops-target-title"),
            html.P(
                "Tabla de ejecución diaria para asignación de responsables y seguimiento.",
                className="card-desc",
            ),
            dash_table.DataTable(
                id="ops-target-table",
                columns=[],
                data=[],
                page_size=12,
                sort_action="native",
                sort_by=[{"column_id": "conversion_score", "direction": "desc"}],
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "padding": "8px 10px",
                    "fontSize": "0.82rem",
                    "fontFamily": "Segoe UI",
                    "textAlign": "left",
                },
                style_header={
                    "backgroundColor": "#1a1a2e",
                    "color": "white",
                    "fontWeight": "700",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
                    {
                        "if": {"column_id": "score_band", "filter_query": "{score_band} = 'ALTO'"},
                        "backgroundColor": "#dcfce7",
                        "color": "#166534",
                        "fontWeight": "700",
                    },
                    {
                        "if": {"column_id": "score_band", "filter_query": "{score_band} = 'MEDIO'"},
                        "backgroundColor": "#fef3c7",
                        "color": "#92400e",
                        "fontWeight": "700",
                    },
                    {
                        "if": {"column_id": "score_band", "filter_query": "{score_band} = 'BAJO'"},
                        "backgroundColor": "#fee2e2",
                        "color": "#991b1b",
                        "fontWeight": "700",
                    },
                ],
            ),
        ], className="dashboard-card"),
    ])


def register_callbacks(app):

    @app.callback(
        Output("ops-main-title", "children"),
        Output("ops-main-subtitle", "children"),
        Output("ops-threshold-label", "children"),
        Output("ops-topn-label", "children"),
        Output("ops-dist-title", "children"),
        Output("ops-dist-desc", "children"),
        Output("ops-kpis", "children"),
        Output("ops-model-info", "children"),
        Output("ops-score-dist", "figure"),
        Output("ops-target-title", "children"),
        Output("ops-target-table", "columns"),
        Output("ops-target-table", "data"),
        Input("ops-threshold", "value"),
        Input("ops-topn", "value"),
        Input("global-lang", "value"),
    )
    def update_ops(threshold, topn, lang):
        lang = normalize_lang(lang)
        df = get_campaign_scores()
        if df.empty:
            empty = _empty_fig(tr("Datos no disponibles", lang))
            return (
                tr("🛠️ Vista Operativa", lang),
                tr("Ejecución diaria de campañas: priorización de clientes, monitoreo del flujo operativo y segmentación accionable.", lang),
                tr("Umbral operativo:", lang),
                tr("Top usuarios a mostrar:", lang),
                tr("Distribución del Score de Conversión", lang),
                tr("Visualiza el corte dinámico del umbral y cuántos usuarios quedan como objetivo de campaña.", lang),
                [],
                tr("Modelo no disponible", lang),
                empty,
                tr("Usuarios Objetivo Filtrados por Umbral", lang) + " (0)",
                [],
                [],
            )

        cls_res = get_classification_results()
        model_name = type(cls_res.get("pipe_rf")).__name__ if cls_res.get("pipe_rf") is not None else "No disponible"

        df = df.copy()
        thr = float(threshold if threshold is not None else DEFAULT_THRESHOLD)
        topn_val = int(topn if topn else 50)

        df["is_target"] = df["conversion_score"] >= thr
        df_target = df[df["is_target"]].sort_values("conversion_score", ascending=False)

        n_total = len(df)
        n_target = len(df_target)
        cobertura = (n_target / n_total * 100.0) if n_total else 0.0
        score_medio_target = float(df_target["conversion_score"].mean()) if n_target else 0.0
        potencial_spend = (
            float(df_target["total_spent"].sum())
            if (n_target and "total_spent" in df_target.columns)
            else 0.0
        )

        kpis = [
            html.Div([
                html.Div(f"{n_total:,}", className="kpi-value"),
                html.Div(tr("Usuarios evaluados", lang), className="kpi-label"),
            ], className="kpi-card blue"),
            html.Div([
                html.Div(f"{n_target:,}", className="kpi-value"),
                html.Div(tr("Usuarios objetivo", lang), className="kpi-label"),
            ], className="kpi-card green"),
            html.Div([
                html.Div(f"{cobertura:.1f}%", className="kpi-value"),
                html.Div(tr("Cobertura de campaña", lang), className="kpi-label"),
            ], className="kpi-card orange"),
            html.Div([
                html.Div(f"{score_medio_target:.3f}", className="kpi-value"),
                html.Div(tr("Score medio objetivo", lang), className="kpi-label"),
            ], className="kpi-card purple"),
            html.Div([
                html.Div(f"${potencial_spend:,.0f}", className="kpi-value"),
                html.Div(tr("Gasto total objetivo", lang), className="kpi-label"),
            ], className="kpi-card red"),
        ]

        model_info = (
            f"{tr('Modelo utilizado para evaluación y scoring operativo:', lang)} {model_name} "
            f"({tr('clasificación de probabilidad de conversión', lang)})."
        )

        fig_dist = px.histogram(
            df,
            x="conversion_score",
            color="is_target",
            nbins=40,
            barmode="overlay",
            opacity=0.8,
            title=tr("Distribución de Score y Segmentación por Umbral", lang),
            labels={"conversion_score": tr("Score de conversión", lang), "is_target": tr("Objetivo", lang)},
            color_discrete_map={True: "#22c55e", False: "#94a3b8"},
        )
        add_non_overlapping_vline_labels(
            fig_dist,
            lines=[
                {
                    "x": thr,
                    "text": f"{tr('Umbral operativo', lang)}: {thr:.2f}",
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
        )

        df_target = df_target.copy()
        df_target["score_band"] = "BAJO"
        df_target.loc[df_target["conversion_score"] >= 0.70, "score_band"] = "ALTO"
        df_target.loc[
            (df_target["conversion_score"] >= 0.45) & (df_target["conversion_score"] < 0.70),
            "score_band",
        ] = "MEDIO"

        table_cols = [
            "client_id",
            "score_band",
            "conversion_score",
            "score_percentil",
            "gender",
            "subscription_type",
            "payment_method",
            "total_spent",
            "avg_order_value",
            "last_3_month_purchase_freq",
            "total_visits",
            "pages_per_session",
            "support_tickets",
            "acquisition_channel",
            "country",
        ]
        table_cols = [c for c in table_cols if c in df_target.columns]

        table_df = df_target[table_cols].head(topn_val).copy()
        if "score_percentil" in table_df.columns:
            table_df["score_percentil"] = table_df["score_percentil"].round(1)

        columns = []
        for c in table_df.columns:
            col = {
                "name": tr(FRIENDLY_COL_NAMES.get(c, c), lang),
                "id": c,
            }
            if c in COL_TYPES:
                col["type"] = COL_TYPES[c]
            if c in COL_FORMATTERS:
                col["format"] = COL_FORMATTERS[c]
            columns.append(col)

        target_title = f"{tr('Usuarios Objetivo Filtrados por Umbral', lang)} ({n_target:,})"

        return (
            tr("🛠️ Vista Operativa", lang),
            tr("Ejecución diaria de campañas: priorización de clientes, monitoreo del flujo operativo y segmentación accionable.", lang),
            tr("Umbral operativo:", lang),
            tr("Top usuarios a mostrar:", lang),
            tr("Distribución del Score de Conversión", lang),
            tr("Visualiza el corte dinámico del umbral y cuántos usuarios quedan como objetivo de campaña.", lang),
            kpis,
            model_info,
            fig_dist,
            target_title,
            columns,
            table_df.to_dict("records"),
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
