"""
Página 3 – Comparación de Resultados (Sucio vs Limpio)
"""

from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_loader import (
    get_df_encoded,
    get_df_clean,
    get_df_raw,
)

PALETTE = px.colors.qualitative.Set2

# Columnas a comparar en análisis de outliers
COLS_OUTLIERS = [
    "age", "total_spent", "avg_order_value", "lifetime_value",
    "total_visits", "avg_session_time", "pages_per_session",
    "support_tickets", "delivery_delay_days",
]


def layout():
    df_raw = get_df_raw()
    df_clean = get_df_clean()

    n_raw = len(df_raw) if not df_raw.empty else "N/A"
    n_clean = len(df_clean) if not df_clean.empty else "N/A"
    nulos_raw = int(df_raw.isnull().sum().sum()) if not df_raw.empty else 0
    nulos_clean = int(df_clean.isnull().sum().sum()) if not df_clean.empty else 0

    return html.Div([
        # ── Encabezado ────────────────────────────────────────────────────────
        html.Div([
            html.H1("📈 Comparación de Resultados: Sucio vs Limpio"),
            html.P("Impacto del proceso de limpieza en la calidad del dataset. "
                   "Comparativa de nulos, distribuciones y outliers."),
        ], className="page-header"),

        # ── KPIs ──────────────────────────────────────────────────────────────
        html.Div([
            html.Div([html.Div(f"{n_raw:,}" if isinstance(n_raw, int) else n_raw,
                               className="kpi-value"),
                      html.Div("Registros originales", className="kpi-label")],
                     className="kpi-card red"),
            html.Div([html.Div(f"{n_clean:,}" if isinstance(n_clean, int) else n_clean,
                               className="kpi-value"),
                      html.Div("Registros limpios", className="kpi-label")],
                     className="kpi-card green"),
            html.Div([html.Div(f"{nulos_raw:,}", className="kpi-value"),
                      html.Div("Nulos en dataset original", className="kpi-label")],
                     className="kpi-card orange"),
            html.Div([html.Div(str(nulos_clean), className="kpi-value"),
                      html.Div("Nulos tras limpieza", className="kpi-label")],
                     className="kpi-card blue"),
        ], className="kpi-row"),

        html.Div([
            "✅ El proceso de limpieza eliminó todos los valores nulos "
            "y corrigió inconsistencias en variables categóricas, mejorando "
            "significativamente la calidad del dataset para el modelado."
        ], className="insight-box success"),

        # ── Comparación de Nulos ──────────────────────────────────────────────
        html.Div([
            html.H3("Nulos por Columna: Antes vs Después"),
            html.P("Comparativa del conteo de nulos en el dataset original "
                   "y el dataset limpio.", className="card-desc"),
            dcc.Graph(id="comp-nulos-chart"),
        ], className="dashboard-card"),

        # ── Distribuciones Lado a Lado ────────────────────────────────────────
        html.Div([
            html.H3("Distribución de Variable: Antes vs Después"),
            html.P("Observa cómo cambia la distribución de una variable "
                   "tras el tratamiento de outliers y limpieza.",
                   className="card-desc"),
            html.Div([
                html.Span("Variable:", className="control-label"),
                dcc.Dropdown(
                    id="comp-var-col",
                    options=[{"label": c, "value": c}
                              for c in COLS_OUTLIERS if not get_df_raw().empty
                              and c in get_df_raw().columns],
                    value="total_spent",
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
            ], className="control-row"),
            html.Div(className="grid-2", children=[
                dcc.Graph(id="comp-dist-raw"),
                dcc.Graph(id="comp-dist-clean"),
            ]),
        ], className="dashboard-card"),

        # ── Categoría Género ──────────────────────────────────────────────────
        html.Div([
            html.H3("Normalización de Variables Categóricas – Ejemplo: Género"),
            html.P("El dataset original contiene valores inconsistentes (mayúsculas, "
                   "espacios). El dataset limpio unifica la representación.",
                   className="card-desc"),
            html.Div(className="grid-2", children=[
                dcc.Graph(id="comp-gender-raw"),
                dcc.Graph(id="comp-gender-clean"),
            ]),
            html.Div([
                "📌 La columna 'gender' tenía variantes como 'male', 'Male', 'MALE'. "
                "Tras la normalización se consolidan en categorías únicas."
            ], className="insight-box warning"),
        ], className="dashboard-card"),

        # ── Comparación de Estadísticas ───────────────────────────────────────
        html.Div([
            html.H3("Tabla Comparativa de Outliers (Mín / Máx / Media)"),
            html.P("Impacto del tratamiento de outliers por variable.",
                   className="card-desc"),
            html.Div([
                html.Span("Variable:", className="control-label"),
                dcc.Dropdown(
                    id="comp-stats-var",
                    options=[{"label": c, "value": c}
                              for c in COLS_OUTLIERS if not get_df_raw().empty
                              and c in get_df_raw().columns],
                    value="total_spent",
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="comp-stats-chart"),
        ], className="dashboard-card"),
    ])


def register_callbacks(app):

    @app.callback(
        Output("comp-nulos-chart", "figure"),
        Input("comp-var-col", "value"),
    )
    def update_nulos(col):
        df_raw = get_df_raw()
        df_clean = get_df_clean()

        if df_raw.empty or df_clean.empty:
            return _empty_fig("Datos no disponibles")

        # Columnas comunes
        common = [c for c in df_clean.columns if c in df_raw.columns]
        nulos_raw = df_raw[common].isnull().sum().reset_index()
        nulos_raw.columns = ["Columna", "Nulos"]
        nulos_raw["Dataset"] = "Original"

        nulos_clean = df_clean[common].isnull().sum().reset_index()
        nulos_clean.columns = ["Columna", "Nulos"]
        nulos_clean["Dataset"] = "Limpio"

        df_comp = pd.concat([nulos_raw, nulos_clean])
        df_comp = df_comp[df_comp["Nulos"] > 0]

        if df_comp.empty:
            return _empty_fig("No se encontraron diferencias de nulos en columnas comunes.")

        fig = px.bar(
            df_comp, x="Columna", y="Nulos",
            color="Dataset",
            barmode="group",
            title="Nulos por Columna – Original vs Limpio",
            color_discrete_map={"Original": "#ef4444", "Limpio": "#22c55e"},
            text="Nulos",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=100, l=60, r=20),
        )
        return fig

    @app.callback(
        Output("comp-dist-raw", "figure"),
        Output("comp-dist-clean", "figure"),
        Input("comp-var-col", "value"),
    )
    def update_dist(col):
        df_raw = get_df_raw()
        df_clean = get_df_clean()

        if not col:
            return _empty_fig("Selecciona una variable"), _empty_fig("")

        def hist(df, title, color):
            if df.empty or col not in df.columns:
                return _empty_fig("Datos no disponibles")
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            fig = px.histogram(
                serie, nbins=40,
                title=title,
                labels={"value": col, "count": "Frecuencia"},
                color_discrete_sequence=[color],
                opacity=0.85,
            )
            mean_v = serie.mean()
            fig.add_vline(x=mean_v, line_dash="dash", line_color="black",
                          annotation_text=f"Media: {mean_v:.1f}")
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="#fafafa",
                font_family="Segoe UI",
                margin=dict(t=50, b=40, l=50, r=20),
            )
            return fig

        return (
            hist(df_raw, f"'{col}' – Dataset Original", "#ef4444"),
            hist(df_clean, f"'{col}' – Dataset Limpio", "#22c55e"),
        )

    @app.callback(
        Output("comp-gender-raw", "figure"),
        Output("comp-gender-clean", "figure"),
        Input("comp-var-col", "value"),
    )
    def update_gender(col):
        def gender_bar(df, title, color_seq):
            if df.empty or "gender" not in df.columns:
                return _empty_fig("Datos no disponibles")
            conteo = df["gender"].astype(str).value_counts().reset_index()
            conteo.columns = ["gender", "count"]
            fig = px.bar(
                conteo, x="gender", y="count",
                title=title,
                color="gender",
                color_discrete_sequence=color_seq,
                text="count",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="#fafafa",
                showlegend=False, font_family="Segoe UI",
                margin=dict(t=50, b=80, l=60, r=20),
            )
            return fig

        red_pal = ["#ef4444", "#f97316", "#fbbf24", "#f43f5e", "#dc2626"]
        green_pal = ["#22c55e", "#16a34a", "#4ade80", "#15803d", "#86efac"]
        return (
            gender_bar(get_df_raw(), "Género – Dataset Original", red_pal),
            gender_bar(get_df_clean(), "Género – Dataset Limpio", green_pal),
        )

    @app.callback(
        Output("comp-stats-chart", "figure"),
        Input("comp-stats-var", "value"),
    )
    def update_stats_comparison(col):
        df_raw = get_df_raw()
        df_clean = get_df_clean()

        if not col:
            return _empty_fig("Selecciona una variable")

        def stats(df, label):
            if df.empty or col not in df.columns:
                return {}
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return {
                "Dataset": label,
                "Mínimo": float(s.min()),
                "Máximo": float(s.max()),
                "Media": float(s.mean()),
                "Mediana": float(s.median()),
            }

        rows = [stats(df_raw, "Original"), stats(df_clean, "Limpio")]
        rows = [r for r in rows if r]
        if not rows:
            return _empty_fig("Datos no disponibles")

        df_stats = pd.DataFrame(rows)
        df_melt = df_stats.melt(id_vars="Dataset", var_name="Métrica", value_name="Valor")

        fig = px.bar(
            df_melt, x="Métrica", y="Valor",
            color="Dataset", barmode="group",
            title=f"Estadísticas de '{col}': Original vs Limpio",
            color_discrete_map={"Original": "#ef4444", "Limpio": "#22c55e"},
            text_auto=".2f",
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=40, l=60, r=20),
        )
        return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#999"))
    fig.update_layout(paper_bgcolor="white")
    return fig
