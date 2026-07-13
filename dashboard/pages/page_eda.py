"""
Página 1 – Análisis Exploratorio de Datos (EDA)
"""

from dash import dcc, html, Input, Output, callback
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
from dashboard.plot_helpers import (
    add_non_overlapping_vline_labels,
    apply_outside_text_anti_overlap,
)

# ── Paleta de colores ─────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2


def layout(lang=None):
    lang = normalize_lang(lang)
    df = get_df_encoded()
    cols_num_disponibles = [c for c in COLS_NUMERICAS if c in df.columns]
    cols_cat_disponibles = [c for c in COLS_CATEGORICAS if c in df.columns]

    n_filas = len(df)
    n_cols = len(df.columns)
    nulos_total = int(df.isnull().sum().sum())
    n_numericas = len(cols_num_disponibles)

    return html.Div([
        # ── Encabezado ────────────────────────────────────────────────────────
        html.Div([
            html.H1(tr("📊 Análisis Exploratorio de Datos (EDA)", lang)),
            html.P(tr("Exploración inicial del dataset: distribuciones, outliers, "
                      "correlaciones y resumen estadístico de variables clave.", lang)),
        ], className="page-header"),

        # ── KPIs ──────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div(f"{n_filas:,}", className="kpi-value"),
                html.Div(tr("Registros", lang), className="kpi-label"),
            ], className="kpi-card blue"),
            html.Div([
                html.Div(str(n_cols), className="kpi-value"),
                html.Div(tr("Columnas totales", lang), className="kpi-label"),
            ], className="kpi-card green"),
            html.Div([
                html.Div(str(n_numericas), className="kpi-value"),
                html.Div(tr("Variables numéricas", lang), className="kpi-label"),
            ], className="kpi-card purple"),
            html.Div([
                html.Div(str(nulos_total), className="kpi-value"),
                html.Div(tr("Valores nulos", lang), className="kpi-label"),
            ], className="kpi-card orange"),
        ], className="kpi-row"),

        # ── Sección 1: Distribuciones ─────────────────────────────────────────
        html.Div([
            html.H3(tr("Distribuciones de Variables Numéricas", lang)),
            html.P(tr("Selecciona una variable para ver su histograma y estadísticas.", lang),
                   className="card-desc"),
            html.Div([
                html.Span(tr("Variable:", lang), className="control-label"),
                dcc.Dropdown(
                    id="eda-num-col",
                    options=[{"label": c, "value": c} for c in cols_num_disponibles],
                    value=cols_num_disponibles[0] if cols_num_disponibles else None,
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
                html.Span(tr("Bins:", lang), className="control-label"),
                dcc.Slider(
                    id="eda-bins",
                    min=10, max=80, step=5, value=30,
                    marks={10: "10", 40: "40", 80: "80"},
                ),
            ], className="control-row"),
            dcc.Graph(id="eda-histogram"),
        ], className="dashboard-card"),

        # ── Sección 2: Boxplots ───────────────────────────────────────────────
        html.Div([
            html.H3(tr("Boxplots – Inspección de Outliers", lang)),
            html.P(tr("Compara la distribución y los outliers entre variables. "
                      "Selecciona hasta 6.", lang), className="card-desc"),
            html.Div([
                html.Span(tr("Variables:", lang), className="control-label"),
                dcc.Dropdown(
                    id="eda-box-cols",
                    options=[{"label": c, "value": c} for c in cols_num_disponibles],
                    value=cols_num_disponibles[:4],
                    multi=True,
                    style={"minWidth": "350px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="eda-boxplot"),
        ], className="dashboard-card"),

        # ── Sección 3: Correlaciones ──────────────────────────────────────────
        html.Div([
            html.H3(tr("Mapa de Correlación (Spearman)", lang)),
            html.P(tr("Correlaciones entre variables numéricas del dataset limpio "
                      "y codificado.", lang), className="card-desc"),
            dcc.Graph(id="eda-corr-heatmap"),
            html.Div([
                tr("💡 Valores cercanos a +1 o -1 indican relación fuerte. "
                   "Valores cercanos a 0 indican independencia estadística.", lang)
            ], className="insight-box"),
        ], className="dashboard-card"),

        # ── Sección 4: Variables Categóricas ─────────────────────────────────
        html.Div([
            html.H3(tr("Distribución de Variables Categóricas", lang)),
            html.P(tr("Frecuencia de las categorías en las variables seleccionadas.", lang),
                   className="card-desc"),
            html.Div([
                html.Span(tr("Categoría:", lang), className="control-label"),
                dcc.Dropdown(
                    id="eda-cat-col",
                    options=[{"label": c, "value": c} for c in cols_cat_disponibles],
                    value=cols_cat_disponibles[0] if cols_cat_disponibles else None,
                    clearable=False,
                    style={"minWidth": "200px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="eda-cat-bar"),
        ], className="dashboard-card"),

        # ── Sección 5: Tabla de Outliers IQR ─────────────────────────────────
        html.Div([
                 html.H3(tr("Resumen de Outliers por IQR", lang)),
                 html.P(tr("Conteo de valores fuera de los límites [Q1 − 1.5·IQR, Q3 + 1.5·IQR].", lang),
                   className="card-desc"),
            dcc.Graph(id="eda-outlier-bar"),
        ], className="dashboard-card"),

    ])


def register_callbacks(app):

    @app.callback(
        Output("eda-histogram", "figure"),
        Input("eda-num-col", "value"),
        Input("eda-bins", "value"),
        Input("global-lang", "value"),
    )
    def update_histogram(col, bins, lang):
        lang = normalize_lang(lang)
        if not col:
            return go.Figure()
        df = get_df_encoded()
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        media = serie.mean()
        mediana = serie.median()

        fig = px.histogram(
            serie, nbins=bins,
            title=f"{tr('Distribución de', lang)} '{col}'",
            labels={"value": col, "count": tr("Frecuencia", lang)},
            color_discrete_sequence=["#3b82f6"],
            opacity=0.85,
        )
        add_non_overlapping_vline_labels(
            fig,
            lines=[
                {
                    "x": float(media),
                    "text": f"{tr('Media', lang)}: {media:.2f}",
                    "line_color": "#ef4444",
                    "line_dash": "dash",
                },
                {
                    "x": float(mediana),
                    "text": f"{tr('Mediana', lang)}: {mediana:.2f}",
                    "line_color": "#22c55e",
                    "line_dash": "dot",
                },
            ],
            data_min=float(serie.min()),
            data_max=float(serie.max()),
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            margin=dict(t=80, b=40, l=50, r=20),
            font_family="Segoe UI",
        )
        return fig

    @app.callback(
        Output("eda-boxplot", "figure"),
        Input("eda-box-cols", "value"),
        Input("global-lang", "value"),
    )
    def update_boxplot(cols, lang):
        lang = normalize_lang(lang)
        if not cols:
            return go.Figure()
        df = get_df_encoded()
        cols = cols[:6]  # máx 6
        fig = go.Figure()
        for i, c in enumerate(cols):
            serie = pd.to_numeric(df[c], errors="coerce").dropna()
            fig.add_trace(go.Box(
                y=serie, name=c,
                marker_color=PALETTE[i % len(PALETTE)],
                boxpoints="outliers",
                jitter=0.3,
            ))
        fig.update_layout(
            title=tr("Boxplots de Variables Seleccionadas", lang),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            margin=dict(t=50, b=40, l=60, r=20),
            font_family="Segoe UI",
            showlegend=False,
        )
        return fig

    @app.callback(
        Output("eda-corr-heatmap", "figure"),
        Input("eda-num-col", "value"),
        Input("global-lang", "value"),
    )
    def update_corr(_, lang):
        lang = normalize_lang(lang)
        df = get_df_encoded()
        cols = [c for c in COLS_NUMERICAS if c in df.columns]
        corr = df[cols].corr(method="spearman").round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            title=tr("Correlación de Spearman", lang),
        )
        fig.update_layout(
            paper_bgcolor="white",
            margin=dict(t=50, b=40, l=10, r=10),
            font_family="Segoe UI",
            height=520,
        )
        return fig

    @app.callback(
        Output("eda-cat-bar", "figure"),
        Input("eda-cat-col", "value"),
        Input("global-lang", "value"),
    )
    def update_cat_bar(col, lang):
        lang = normalize_lang(lang)
        if not col:
            return go.Figure()
        df = get_df_clean()
        if df.empty or col not in df.columns:
            df = get_df_encoded()
        if col not in df.columns:
            return go.Figure()

        conteo = df[col].astype(str).value_counts().head(15).reset_index()
        conteo.columns = [col, "count"]

        fig = px.bar(
            conteo, x=col, y="count",
            title=f"{tr('Distribución de', lang)} '{col}'",
            labels={col: col, "count": tr("Cantidad", lang)},
            color=col,
            color_discrete_sequence=PALETTE,
            text="count",
        )
        apply_outside_text_anti_overlap(fig, min_top=70)
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            margin=dict(t=50, b=80, l=60, r=20),
            font_family="Segoe UI",
            showlegend=False,
        )
        return fig

    @app.callback(
        Output("eda-outlier-bar", "figure"),
        Input("eda-num-col", "value"),
        Input("global-lang", "value"),
    )
    def update_outliers(_, lang):
        lang = normalize_lang(lang)
        df = get_df_encoded()
        cols = [c for c in COLS_NUMERICAS if c in df.columns]
        rows = []
        for col in cols:
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())
            rows.append({"Variable": col, "Outliers (IQR)": n_out})

        df_out = pd.DataFrame(rows).sort_values("Outliers (IQR)", ascending=False)
        fig = px.bar(
            df_out, x="Variable", y="Outliers (IQR)",
            title=tr("Cantidad de Outliers por Variable (Método IQR)", lang),
            color="Outliers (IQR)",
            color_continuous_scale="Oranges",
            text="Outliers (IQR)",
        )
        apply_outside_text_anti_overlap(fig, min_top=70)
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            margin=dict(t=50, b=80, l=60, r=20),
            font_family="Segoe UI",
            coloraxis_showscale=False,
        )
        return fig
