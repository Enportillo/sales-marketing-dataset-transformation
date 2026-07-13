"""
Página 4 – Modelado: Clustering KMeans + Modelos Supervisados
"""

from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_loader import (
    get_df_encoded,
    get_cluster_results,
    COLS_COMPORTAMIENTO,
)
from dashboard.i18n import tr, normalize_lang
from dashboard.plot_helpers import add_hline_label

PALETTE = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]
CLUSTER_LABELS = {0: "🟢 Activos", 1: "🔵 Regulares", 2: "🟠 Esporádicos / En Riesgo"}


def layout(lang=None):
    lang = normalize_lang(lang)
    return html.Div([
        # ── Encabezado ────────────────────────────────────────────────────────
        html.Div([
            html.H1(tr("🤖 Modelado – Clustering KMeans", lang)),
            html.P(tr("Segmentación de clientes mediante KMeans (K=3) sobre variables "
                      "de comportamiento. Validación con método del codo y Silhouette Score.", lang)),
        ], className="page-header"),

        # ── Insight de negocio ────────────────────────────────────────────────
        html.Div([
            tr(
                "📌 Se utilizaron 6 variables de comportamiento: gasto total, ticket promedio, "
                "frecuencia de compra (3 meses), visitas, páginas por sesión y tickets de soporte. "
                "Se eligió K=3 por su interpretabilidad comercial.",
                lang,
            )
        ], className="insight-box"),

        # ── Sección 1: Validación matemática ──────────────────────────────────
        html.Div([
                 html.H3(tr("Validación Matemática del Número de Clústeres", lang)),
                 html.P(tr("Método del Codo (Inercia) y Silhouette Score para K = 2..6.", lang),
                   className="card-desc"),
            html.Div(className="grid-2", children=[
                dcc.Graph(id="model-elbow"),
                dcc.Graph(id="model-silhouette"),
            ]),
            html.Div([
                tr(
                    "💡 El codo sugiere mejora decreciente a partir de K=4. "
                    "Silhouette favorece K=2 estadísticamente. "
                    "Se eligió K=3 por accionabilidad comercial: permite diseñar "
                    "estrategias diferenciadas sin perder simplicidad.",
                    lang,
                )
            ], className="insight-box warning"),
        ], className="dashboard-card"),

        # ── Sección 2: Visualización PCA ──────────────────────────────────────
        html.Div([
                 html.H3(tr("Visualización de Clústeres (PCA 2D)", lang)),
                 html.P(tr("Proyección de los clientes en 2 componentes principales "
                     "para comunicar los perfiles de forma visual.", lang),
                   className="card-desc"),
            dcc.Graph(id="model-pca-scatter"),
            html.Div([
                tr(
                    "🔍 La transición gradual entre perfiles (sin fronteras rígidas) "
                    "sugiere que pequeños estímulos comerciales pueden mover clientes entre segmentos.",
                    lang,
                )
            ], className="insight-box"),
        ], className="dashboard-card"),

        # ── Sección 3: Perfiles de clústeres ─────────────────────────────────
        html.Div([
            html.H3(tr("Perfilamiento de Clústeres (Centroides)", lang)),
            html.P(tr("Media de cada variable de comportamiento por perfil de cliente.", lang),
                   className="card-desc"),
            html.Div([
                html.Span(tr("Variable a visualizar:", lang), className="control-label"),
                dcc.Dropdown(
                    id="model-centroid-var",
                    options=[{"label": c, "value": c}
                              for c in COLS_COMPORTAMIENTO],
                    value=COLS_COMPORTAMIENTO,
                    multi=True,
                    style={"minWidth": "300px"},
                ),
            ], className="control-row"),
            dcc.Graph(id="model-centroid-chart"),
        ], className="dashboard-card"),

        # ── Sección 4: Radar de perfiles ──────────────────────────────────────
        html.Div([
                 html.H3(tr("Radar – Comparación de Perfiles", lang)),
                 html.P(tr("Vista de araña normalizada de los 3 perfiles de cliente.", lang),
                   className="card-desc"),
            dcc.Graph(id="model-radar"),
            html.Div(className="grid-3", children=[
                html.Div([
                    html.Strong("🟢 Clúster 0 – Activos"),
                    html.P("Alta frecuencia de compra y flujo constante. "
                           "Son el motor de ventas.", style={"fontSize": "0.83rem"}),
                ], className="dashboard-card", style={"borderTop": "4px solid #2ca02c"}),
                html.Div([
                    html.Strong("🔵 Clúster 1 – Regulares"),
                    html.P("Comportamiento intermedio. Gran potencial de crecimiento "
                           "con acciones de retención.", style={"fontSize": "0.83rem"}),
                ], className="dashboard-card", style={"borderTop": "4px solid #1f77b4"}),
                html.Div([
                    html.Strong("🟠 Clúster 2 – Esporádicos"),
                    html.P("Baja frecuencia, pero buen gasto cuando compran. "
                           "Foco de campañas de reactivación.", style={"fontSize": "0.83rem"}),
                ], className="dashboard-card", style={"borderTop": "4px solid #ff7f0e"}),
            ]),
        ], className="dashboard-card"),

        # ── Sección 5: Distribución de clientes ───────────────────────────────
        html.Div([
                 html.H3(tr("Distribución de Clientes por Perfil", lang)),
                 html.P(tr("Tamaño de cada segmento para dimensionar campañas.", lang),
                   className="card-desc"),
            dcc.Graph(id="model-cluster-dist"),
        ], className="dashboard-card"),
    ])


def register_callbacks(app):

    @app.callback(
        Output("model-elbow", "figure"),
        Output("model-silhouette", "figure"),
        Input("model-centroid-var", "value"),
        Input("global-lang", "value"),
    )
    def update_validation(cols, lang):
        lang = normalize_lang(lang)
        res = get_cluster_results()
        k_range = res["k_range"]
        inertias = res["inertias"]
        silhouettes = res["silhouettes"]

        # Elbow
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_range, y=inertias,
            mode="lines+markers",
            marker=dict(size=9, color="#3b82f6", symbol="circle"),
            line=dict(color="#3b82f6", width=2, dash="dash"),
        ))
        fig_elbow.update_layout(
            title=tr("Método del Codo (Inercia)", lang),
            xaxis_title=tr("Número de Clústeres (K)", lang),
            yaxis_title=tr("Inercia", lang),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=40, l=60, r=20),
        )

        # Silhouette
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_range, y=silhouettes,
            mode="lines+markers",
            marker=dict(size=9, color="#22c55e", symbol="square"),
            line=dict(color="#22c55e", width=2),
        ))
        add_hline_label(
            fig_sil,
            y=float(silhouettes[1]),
            text=tr("K=3 elegido", lang),
            line_color="#f59e0b",
            line_dash="dot",
        )
        fig_sil.update_layout(
            title=tr("Validación Silhouette Score", lang),
            xaxis_title=tr("Número de Clústeres (K)", lang),
            yaxis_title=tr("Silhouette Score", lang),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=65, b=40, l=60, r=20),
        )
        return fig_elbow, fig_sil

    @app.callback(
        Output("model-pca-scatter", "figure"),
        Input("model-centroid-var", "value"),
        Input("global-lang", "value"),
    )
    def update_pca(cols, lang):
        lang = normalize_lang(lang)
        res = get_cluster_results()
        X_pca = res["X_pca"]
        labels = res["labels"]
        var = res["pca_var"]

        df_pca = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": [CLUSTER_LABELS.get(l, str(l)) for l in labels],
        })

        fig = px.scatter(
            df_pca, x="PC1", y="PC2",
            color="Cluster",
            color_discrete_sequence=PALETTE,
            opacity=0.6,
            title=tr("Visualización de Clústeres de Clientes (PCA 2D)", lang),
            labels={
                "PC1": f"{tr('Componente 1', lang)} ({var[0]*100:.1f}% {tr('varianza', lang)})",
                "PC2": f"{tr('Componente 2', lang)} ({var[1]*100:.1f}% {tr('varianza', lang)})",
            },
        )
        fig.update_traces(marker_size=5)
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=60, b=40, l=60, r=20),
            legend_title=tr("Perfil (Clúster)", lang),
            height=480,
        )
        return fig

    @app.callback(
        Output("model-centroid-chart", "figure"),
        Input("model-centroid-var", "value"),
        Input("global-lang", "value"),
    )
    def update_centroids(cols, lang):
        lang = normalize_lang(lang)
        if not cols:
            return go.Figure()
        res = get_cluster_results()
        centroids = res["centroids"]
        cols_disp = [c for c in cols if c in centroids.columns]
        if not cols_disp:
            return go.Figure()

        df_cent = centroids[cols_disp].reset_index()
        df_melt = df_cent.melt(id_vars="cluster", var_name="Variable", value_name="Media")
        df_melt["Cluster"] = df_melt["cluster"].map(
            lambda x: CLUSTER_LABELS.get(x, f"Clúster {x}")
        )

        fig = px.bar(
            df_melt, x="Variable", y="Media",
            color="Cluster",
            barmode="group",
            title=tr("Centroides por Variable de Comportamiento", lang),
            color_discrete_sequence=PALETTE,
            text_auto=".2f",
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font_family="Segoe UI",
            margin=dict(t=50, b=80, l=60, r=20),
            legend_title=tr("Perfil", lang),
        )
        return fig

    @app.callback(
        Output("model-radar", "figure"),
        Input("model-centroid-var", "value"),
        Input("global-lang", "value"),
    )
    def update_radar(cols, lang):
        lang = normalize_lang(lang)
        res = get_cluster_results()
        centroids = res["centroids"]
        cols = res["cols"]

        # Normalizar 0-1 para el radar
        c_norm = centroids[cols].copy()
        for col in cols:
            mn, mx = c_norm[col].min(), c_norm[col].max()
            if mx > mn:
                c_norm[col] = (c_norm[col] - mn) / (mx - mn)

        fig = go.Figure()
        for i, (idx, row) in enumerate(c_norm.iterrows()):
            valores = list(row.values)
            valores.append(valores[0])  # cerrar radar
            categorias = cols + [cols[0]]
            fig.add_trace(go.Scatterpolar(
                r=valores, theta=categorias,
                fill="toself", name=CLUSTER_LABELS.get(idx, f"Clúster {idx}"),
                line_color=PALETTE[i],
                fillcolor=PALETTE[i],
                opacity=0.4,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=tr("Radar de Perfiles de Cliente (Normalizado)", lang),
            paper_bgcolor="white",
            font_family="Segoe UI",
            margin=dict(t=60, b=40, l=60, r=60),
            height=450,
            legend_title=tr("Perfil", lang),
        )
        return fig

    @app.callback(
        Output("model-cluster-dist", "figure"),
        Input("model-centroid-var", "value"),
        Input("global-lang", "value"),
    )
    def update_dist(cols, lang):
        lang = normalize_lang(lang)
        res = get_cluster_results()
        centroids = res["centroids"]

        if "n_clientes" not in centroids.columns:
            labels_arr = res["labels"]
            counts = pd.Series(labels_arr).value_counts().sort_index()
        else:
            counts = centroids["n_clientes"].sort_index()

        df_dist = pd.DataFrame({
            "Perfil": [CLUSTER_LABELS.get(i, f"Clúster {i}") for i in counts.index],
            "Clientes": counts.values,
        })

        fig = px.pie(
            df_dist, values="Clientes", names="Perfil",
            title=tr("Distribución de Clientes por Perfil", lang),
            color_discrete_sequence=PALETTE,
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label+value")
        fig.update_layout(
            paper_bgcolor="white",
            font_family="Segoe UI",
            margin=dict(t=60, b=40),
        )
        return fig
