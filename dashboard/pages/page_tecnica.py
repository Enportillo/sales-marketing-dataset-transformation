"""
Vista Técnica – Arquitectura y Analítica Avanzada
"""

from dash import html, Input, Output
from dashboard.i18n import tr, normalize_lang


def _build_content(lang: str):
    cards = [
        {
            "title": "EDA – Análisis Exploratorio",
            "desc": "Distribuciones, correlaciones, calidad estadística y patrones base.",
            "href": "/eda",
            "badge": "Datos",
            "color": "#3b82f6",
        },
        {
            "title": "Transformación de Datos",
            "desc": "Limpieza, codificación y reglas del pipeline de preparación.",
            "href": "/transformacion",
            "badge": "ETL",
            "color": "#f59e0b",
        },
        {
            "title": "Comparación Sucio vs Limpio",
            "desc": "Impacto técnico del preprocesamiento en nulos y distribuciones.",
            "href": "/comparacion",
            "badge": "Calidad",
            "color": "#22c55e",
        },
        {
            "title": "Modelado",
            "desc": "Segmentación, validación de clústeres y comportamiento de modelos.",
            "href": "/modelado",
            "badge": "ML",
            "color": "#8b5cf6",
        },
        {
            "title": "Evaluación de Modelos",
            "desc": "Matrices, curvas ROC, métricas de clasificación y regresión.",
            "href": "/evaluacion",
            "badge": "Validación",
            "color": "#ef4444",
        },
        {
            "title": "Optimización de Hiperparámetros",
            "desc": "Comparativa de versiones, búsqueda de parámetros y performance.",
            "href": "/optimizacion",
            "badge": "Tuning",
            "color": "#14b8a6",
        },
    ]

    return [
        html.Div([
            html.H1(tr("🧩 Vista Técnica", lang)),
            html.P(
                tr(
                    "Acceso a módulos de ingeniería de datos y machine learning para análisis "
                    "en profundidad de arquitectura, rendimiento y calidad de modelo.",
                    lang,
                )
            ),
        ], className="page-header"),

        html.Div([
            tr("Esta vista organiza el detalle técnico por dominios. Usa los accesos rápidos para navegar a cada módulo especializado.", lang)
        ], className="insight-box"),

        html.Div([
            html.A([
                html.Div([
                    html.Span(tr(card["badge"], lang), className="badge-rf", style={"background": card["color"]}),
                    html.H3(tr(card["title"], lang), style={"marginTop": "10px"}),
                    html.P(tr(card["desc"], lang), className="card-desc", style={"marginTop": "8px"}),
                    html.Div(tr("Ir al módulo →", lang), style={"fontWeight": "700", "fontSize": "0.85rem", "color": card["color"]}),
                ], className="dashboard-card", style={"borderTop": f"4px solid {card['color']}", "height": "100%"}),
            ], href=card["href"], style={"textDecoration": "none"})
            for card in cards
        ], className="grid-2"),
    ]


def layout():
    return html.Div(id="tech-page-root", children=_build_content("es"))


def register_callbacks(app):
    @app.callback(
        Output("tech-page-root", "children"),
        Input("global-lang", "value"),
    )
    def update_tech(lang):
        return _build_content(normalize_lang(lang))
