"""
Página 2 – Transformación de Datos
"""

from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_loader import (
    get_df_encoded,
    get_df_clean,
    COLS_NUMERICAS,
    COLS_CATEGORICAS,
)
from dashboard.i18n import tr, normalize_lang
from dashboard.plot_helpers import apply_outside_text_anti_overlap

PALETTE = px.colors.qualitative.Set2


def layout(lang=None):
    lang = normalize_lang(lang)
    df_enc = get_df_encoded()
    df_clean = get_df_clean()

    # Estadísticas rápidas
    n_enc = len(df_enc)
    n_cols_enc = len(df_enc.columns)
    nulos_enc = int(df_enc.isnull().sum().sum())

    # Dtypes del dataset codificado
    dtypes = df_enc.dtypes.reset_index()
    dtypes.columns = ["Columna", "Tipo"]
    dtypes["Tipo"] = dtypes["Tipo"].astype(str)
    dtypes_int = (dtypes["Tipo"] == "int64").sum()
    dtypes_float = (dtypes["Tipo"] == "float64").sum()
    dtypes_obj = (dtypes["Tipo"] == "object").sum()

    return html.Div([
        # ── Encabezado ────────────────────────────────────────────────────────
        html.Div([
            html.H1(tr("🔧 Transformación de Datos", lang)),
            html.P(tr("Proceso de limpieza, codificación y preparación del dataset "
                      "para el modelado ML.", lang)),
        ], className="page-header"),

        # ── KPIs ──────────────────────────────────────────────────────────────
        html.Div([
            html.Div([html.Div(f"{n_enc:,}", className="kpi-value"),
                      html.Div(tr("Registros finales", lang), className="kpi-label")],
                     className="kpi-card green"),
            html.Div([html.Div(str(n_cols_enc), className="kpi-value"),
                      html.Div(tr("Columnas finales", lang), className="kpi-label")],
                     className="kpi-card blue"),
            html.Div([html.Div(str(nulos_enc), className="kpi-value"),
                      html.Div(tr("Nulos en dataset final", lang), className="kpi-label")],
                     className="kpi-card orange"),
            html.Div([html.Div(str(dtypes_int + dtypes_float), className="kpi-value"),
                      html.Div(tr("Columnas numéricas", lang), className="kpi-label")],
                     className="kpi-card purple"),
        ], className="kpi-row"),

        # ── Pipeline de transformación ────────────────────────────────────────
        html.Div([
                 html.H3(tr("Pipeline de Transformación", lang)),
                 html.P(tr("Pasos aplicados al dataset original para obtener el dataset limpio codificado.", lang),
                   className="card-desc"),
            _render_pipeline_steps(lang),
        ], className="dashboard-card"),

        # ── Tipos de datos ─────────────────────────────────────────────────────
        html.Div([
                 html.H3(tr("Composición de Tipos de Datos (Dataset Codificado)", lang)),
                 html.P(tr("Distribución de tipos de datos en el dataset final.", lang),
                   className="card-desc"),
        html.Div(className="grid-2", children=[
                dcc.Graph(id="trans-dtypes-pie"),
                dcc.Graph(id="trans-dtypes-bar"),
            ]),
        ], className="dashboard-card"),

        # ── Estadísticas descriptivas ─────────────────────────────────────────
        html.Div([
            html.H3(tr("Estadísticas Descriptivas – Dataset Codificado", lang)),
            html.P(tr("Resumen de media, desviación estándar, mínimo y máximo por variable.", lang),
                   className="card-desc"),
            html.Div([
                html.Span(tr("Variable:", lang), className="control-label"),
                dcc.Dropdown(
                    id="trans-num-col",
                    options=[{"label": c, "value": c}
                              for c in COLS_NUMERICAS if c in df_enc.columns],
                    value=[c for c in COLS_NUMERICAS if c in df_enc.columns][:5],
                    multi=True,
                    style={"minWidth": "300px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="trans-stats-chart"),
        ], className="dashboard-card"),

        # ── Vista de muestra del dataset ──────────────────────────────────────
        html.Div([
                 html.H3(tr("Muestra del Dataset Codificado (primeras 10 filas)", lang)),
                 html.P(tr("Vista rápida del dataset final listo para modelado.", lang),
                   className="card-desc"),
            _render_sample_table(df_enc),
        ], className="dashboard-card"),

        # ── Codificación de categóricas ───────────────────────────────────────
        html.Div([
            html.H3(tr("Codificación de Variables Categóricas", lang)),
            html.P(tr("Variables codificadas numéricamente para el modelado supervisado.", lang),
                   className="card-desc"),
            html.Div([
                html.Span(tr("Variable a comparar:", lang), className="control-label"),
                dcc.Dropdown(
                    id="trans-enc-col",
                    options=[{"label": c, "value": c}
                              for c in COLS_CATEGORICAS if c in df_enc.columns],
                    value="subscription_type" if "subscription_type" in df_enc.columns else
                          ([c for c in COLS_CATEGORICAS if c in df_enc.columns] or [None])[0],
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="trans-enc-chart"),
        ], className="dashboard-card"),
    ])


def _render_pipeline_steps(lang=None):
    lang = normalize_lang(lang)
    steps = [
        ("1️⃣", "Carga del dataset bruto",
         "Lectura del archivo Excel con datos sucios (nulls, outliers, inconsistencias)."),
        ("2️⃣", "Imputación de nulos",
         "Valores nulos en variables numéricas imputados con la mediana. "
         "Categóricas con el valor más frecuente."),
        ("3️⃣", "Normalización de categóricas",
         "Unificación de capitalización y corrección de strings mal formateados "
         "(ej: 'male', 'Male', 'MALE' → 'Male')."),
        ("4️⃣", "Tratamiento de outliers",
         "Outliers detectados por IQR recortados a los límites [Q1 − 1.5·IQR, Q3 + 1.5·IQR]."),
        ("5️⃣", "Codificación Label Encoding",
         "Variables categóricas convertidas a enteros (0, 1, 2, …) para uso en scikit-learn."),
        ("6️⃣", "Exportación del dataset final",
         "Guardado en Sales_Marketing_Clean_(Codificado).csv listo para modelado."),
    ]
    items = []
    for icon, title, desc in steps:
        items.append(html.Div([
            html.Div(icon, style={"fontSize": "1.5rem", "marginRight": "12px"}),
            html.Div([
                html.Strong(tr(title, lang), style={"display": "block", "marginBottom": "2px"}),
                html.Span(tr(desc, lang), style={"fontSize": "0.82rem", "color": "#555"}),
            ])
        ], style={
            "display": "flex", "alignItems": "flex-start",
            "background": "#f8f9fa", "borderRadius": "8px",
            "padding": "12px 16px", "marginBottom": "8px",
            "borderLeft": "3px solid #3b82f6",
        }))
    return html.Div(items)


def _render_sample_table(df: pd.DataFrame):
    sample = df.head(10)
    cols = [{"name": c, "id": c} for c in sample.columns]
    data = sample.to_dict("records")
    return dash_table.DataTable(
        columns=cols,
        data=data,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px 12px", "fontSize": "0.8rem",
                    "fontFamily": "Segoe UI", "textAlign": "left"},
        style_header={"backgroundColor": "#1a1a2e", "color": "white",
                      "fontWeight": "700", "fontSize": "0.78rem"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
        ],
        page_size=10,
    )


def register_callbacks(app):

    @app.callback(
        Output("trans-dtypes-pie", "figure"),
        Output("trans-dtypes-bar", "figure"),
        Input("trans-num-col", "value"),
        Input("global-lang", "value"),
    )
    def update_dtypes(_cols, lang):
        lang = normalize_lang(lang)
        df = get_df_encoded()
        dtypes = df.dtypes.astype(str).value_counts().reset_index()
        dtypes.columns = ["Tipo", "Cantidad"]
        pie = px.pie(dtypes, values="Cantidad", names="Tipo",
                     title=tr("Distribución de Tipos", lang),
                     color_discrete_sequence=PALETTE, hole=0.4)
        pie.update_layout(paper_bgcolor="white", font_family="Segoe UI",
                          margin=dict(t=50, b=20))

        bar = px.bar(dtypes, x="Tipo", y="Cantidad",
                     title=tr("Cantidad de Columnas por Tipo", lang),
                     color="Tipo", color_discrete_sequence=PALETTE,
                     text="Cantidad")
        apply_outside_text_anti_overlap(bar, min_top=70)
        bar.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa",
                          showlegend=False, font_family="Segoe UI",
                          margin=dict(t=50, b=40))
        return pie, bar

    @app.callback(
        Output("trans-stats-chart", "figure"),
        Input("trans-num-col", "value"),
        Input("global-lang", "value"),
    )
    def update_stats(cols, lang):
        lang = normalize_lang(lang)
        if not cols:
            return go.Figure()
        df = get_df_encoded()
        cols = cols[:8]
        desc = df[cols].describe().T[["mean", "std", "min", "max"]].reset_index()
        desc.columns = ["Variable", "Media", "Std", "Mínimo", "Máximo"]

        fig = go.Figure()
        for metric, color in [("Media", "#3b82f6"), ("Std", "#f59e0b"),
                               ("Mínimo", "#22c55e"), ("Máximo", "#ef4444")]:
            fig.add_trace(go.Bar(
                name=metric, x=desc["Variable"], y=desc[metric],
                marker_color=color,
            ))
        fig.update_layout(
            barmode="group",
            title=tr("Estadísticas Descriptivas por Variable", lang),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=80, l=60, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        return fig

    @app.callback(
        Output("trans-enc-chart", "figure"),
        Input("trans-enc-col", "value"),
        Input("global-lang", "value"),
    )
    def update_encoding(col, lang):
        lang = normalize_lang(lang)
        if not col:
            return go.Figure()
        df = get_df_encoded()
        if col not in df.columns:
            return go.Figure()

        conteo = df[col].value_counts().reset_index()
        conteo.columns = [col, "count"]
        fig = px.bar(
            conteo, x=col, y="count",
            title=f"{tr('Distribución de', lang)} '{col}' ({tr('Valor Codificado', lang)})",
            color=col,
            color_discrete_sequence=PALETTE,
            text="count",
            labels={col: f"{col} ({tr('código', lang)})", "count": tr("Cantidad", lang)},
        )
        apply_outside_text_anti_overlap(fig, min_top=70)
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            showlegend=False, font_family="Segoe UI",
            margin=dict(t=50, b=60, l=60, r=20),
        )
        return fig
