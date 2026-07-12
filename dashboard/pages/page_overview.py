"""
Página 0 – Overview de Conversión a Nivel Usuario
"""

from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from dash.dash_table.Format import Format, Group, Scheme, Symbol

from dashboard.data_loader import get_campaign_scores, get_classification_results
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
            html.H1("🎯 Overview – Campaña de Conversión"),
            html.P(
                "Identificación dinámica de usuarios objetivo según probabilidad "
                "de conversión. Ajusta el umbral para cambiar cobertura y foco comercial."
            ),
        ], className="page-header"),

        html.Div([
            html.Div([
                html.Span("Umbral de conversión:", className="control-label"),
                dcc.Slider(
                    id="ov-threshold",
                    min=0.10,
                    max=0.95,
                    step=0.01,
                    value=DEFAULT_THRESHOLD,
                    marks={v: f"{int(v * 100)}%" for v in [0.10, 0.25, 0.34, 0.50, 0.75, 0.90]},
                ),
            ], style={"flex": "2"}),
            html.Div([
                html.Span("Top usuarios a mostrar:", className="control-label"),
                dcc.Dropdown(
                    id="ov-topn",
                    options=[{"label": str(n), "value": n} for n in [10, 25, 50, 100, 250]],
                    value=50,
                    clearable=False,
                    style={"minWidth": "180px"},
                ),
            ], style={"flex": "1", "minWidth": "220px"}),
        ], className="dashboard-card control-row"),

        html.Div(id="ov-kpis", className="kpi-row"),

        html.Div(id="ov-model-info", className="insight-box"),

        html.Div([
            html.H3("Distribución del Score de Conversión"),
            html.P(
                "Visualiza el corte dinámico del umbral y cuántos usuarios quedan como objetivo.",
                className="card-desc",
            ),
            dcc.Graph(id="ov-score-dist"),
        ], className="dashboard-card"),

        html.Div(className="grid-2", children=[
            html.Div([
                html.H3("Tendencia por Canal de Adquisición"),
                html.P("Score promedio de conversión por canal.", className="card-desc"),
                dcc.Graph(id="ov-channel-trend"),
            ], className="dashboard-card"),
            html.Div([
                html.H3("Tendencia por País"),
                html.P("Top países por score promedio de conversión.", className="card-desc"),
                dcc.Graph(id="ov-country-trend"),
            ], className="dashboard-card"),
        ]),

        html.Div([
            html.H3(id="ov-target-title"),
            html.P(
                "La tabla se actualiza automáticamente al mover el umbral. "
                "Puedes ordenar y filtrar dentro de la grilla.",
                className="card-desc",
            ),
            dash_table.DataTable(
                id="ov-target-table",
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
        Output("ov-kpis", "children"),
        Output("ov-model-info", "children"),
        Output("ov-score-dist", "figure"),
        Output("ov-channel-trend", "figure"),
        Output("ov-country-trend", "figure"),
        Output("ov-target-title", "children"),
        Output("ov-target-table", "columns"),
        Output("ov-target-table", "data"),
        Input("ov-threshold", "value"),
        Input("ov-topn", "value"),
    )
    def update_overview(threshold, topn):
        df = get_campaign_scores()
        if df.empty:
            empty = _empty_fig("Datos no disponibles")
            return [], "Modelo no disponible", empty, empty, empty, "Usuarios Objetivo Filtrados por Umbral (0)", [], []

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

        model_info = (
            f"Modelo utilizado para evaluación y scoring: {model_name} "
            "(clasificación de probabilidad de conversión)."
        )

        kpis = [
            html.Div([
                html.Div(f"{n_total:,}", className="kpi-value"),
                html.Div("Usuarios evaluados", className="kpi-label"),
            ], className="kpi-card blue"),
            html.Div([
                html.Div(f"{n_target:,}", className="kpi-value"),
                html.Div("Usuarios objetivo", className="kpi-label"),
            ], className="kpi-card green"),
            html.Div([
                html.Div(f"{cobertura:.1f}%", className="kpi-value"),
                html.Div("Cobertura de campaña", className="kpi-label"),
            ], className="kpi-card orange"),
            html.Div([
                html.Div(f"{score_medio_target:.3f}", className="kpi-value"),
                html.Div("Score medio objetivo", className="kpi-label"),
            ], className="kpi-card purple"),
            html.Div([
                html.Div(f"${potencial_spend:,.0f}", className="kpi-value"),
                html.Div("Total spent objetivo", className="kpi-label"),
            ], className="kpi-card red"),
        ]

        fig_dist = px.histogram(
            df,
            x="conversion_score",
            color="is_target",
            nbins=40,
            barmode="overlay",
            opacity=0.8,
            title="Distribución de Score y Segmentación por Umbral",
            labels={"conversion_score": "Score de conversión", "is_target": "Objetivo"},
            color_discrete_map={True: "#22c55e", False: "#94a3b8"},
        )
        add_non_overlapping_vline_labels(
            fig_dist,
            lines=[
                {
                    "x": thr,
                    "text": f"Umbral: {thr:.2f}",
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

        if "acquisition_channel" in df.columns:
            by_channel = (
                df.groupby("acquisition_channel", as_index=False)["conversion_score"]
                .mean()
                .sort_values("conversion_score", ascending=False)
            )
            fig_channel = px.bar(
                by_channel,
                x="acquisition_channel",
                y="conversion_score",
                title="Score Promedio por Canal",
                color_discrete_sequence=["#3b82f6"],
                text_auto=".3f",
            )
            fig_channel.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="#fafafa",
                font_family="Segoe UI",
                margin=dict(t=50, b=60, l=60, r=20),
            )
        else:
            fig_channel = _empty_fig("Columna acquisition_channel no disponible")

        if "country" in df.columns:
            by_country = (
                df.groupby("country", as_index=False)["conversion_score"]
                .mean()
                .sort_values("conversion_score", ascending=False)
                .head(15)
            )
            fig_country = px.bar(
                by_country,
                x="country",
                y="conversion_score",
                title="Top 15 Países por Score Promedio",
                color_discrete_sequence=["#10b981"],
                text_auto=".3f",
            )
            fig_country.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="#fafafa",
                font_family="Segoe UI",
                margin=dict(t=50, b=60, l=60, r=20),
            )
        else:
            fig_country = _empty_fig("Columna country no disponible")

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
                "name": FRIENDLY_COL_NAMES.get(c, c),
                "id": c,
            }
            if c in COL_TYPES:
                col["type"] = COL_TYPES[c]
            if c in COL_FORMATTERS:
                col["format"] = COL_FORMATTERS[c]
            columns.append(col)
        data = table_df.to_dict("records")

        target_title = f"Usuarios Objetivo Filtrados por Umbral ({n_target:,})"

        return kpis, model_info, fig_dist, fig_channel, fig_country, target_title, columns, data


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
