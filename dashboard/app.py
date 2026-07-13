"""
Dashboard Interactivo – Sales & Marketing ML
=============================================
Aplicación Dash con menú lateral izquierdo y 6 páginas:
  1. EDA – Análisis Exploratorio
  2. Transformación de Datos
  3. Comparación de Resultados
  4. Modelado (KMeans + Supervisado)
  5. Evaluación de Modelos
  6. Optimización de Hiperparámetros

Uso:
    cd dashboard
    python app.py
    # Abre http://127.0.0.1:8050 en el navegador
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DASH_DIR = os.path.dirname(os.path.abspath(__file__))
if DASH_DIR not in sys.path:
    sys.path.insert(0, DASH_DIR)

import dash
from dash import dcc, html, Input, Output, State
from dashboard.i18n import LANG_OPTIONS, DEFAULT_LANG

# ── Importar páginas ──────────────────────────────────────────────────────────
from dashboard.pages import (
    page_ejecutiva,
    page_operativa,
    page_tecnica,
    page_overview,
    page_eda,
    page_transformacion,
    page_comparacion,
    page_modelado,
    page_evaluacion,
    page_optimizacion,
)

# ── Inicializar app ───────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Sales & Marketing Dashboard",
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(DASH_DIR, "assets"),
)
server = app.server  # Para despliegue con Gunicorn/Waitress

# ── Registrar callbacks de cada página ───────────────────────────────────────
page_ejecutiva.register_callbacks(app)
page_operativa.register_callbacks(app)
page_tecnica.register_callbacks(app)
page_overview.register_callbacks(app)
page_eda.register_callbacks(app)
page_transformacion.register_callbacks(app)
page_comparacion.register_callbacks(app)
page_modelado.register_callbacks(app)
page_evaluacion.register_callbacks(app)
page_optimizacion.register_callbacks(app)

# ── Definición del sidebar ────────────────────────────────────────────────────
NAV_ITEMS = [
    ("/ejecutiva",     "🏛️", "Vista Ejecutiva",         "Decisiones estratégicas"),
    ("/operativa",     "🛠️", "Vista Operativa",         "Ejecución de campañas"),
    ("/tecnica",       "🧩", "Vista Técnica",           "Arquitectura y detalle ML"),
]

sidebar = html.Div([
    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.H2("Sales & Marketing\nML Dashboard"),
        html.P("Proyecto de Ciencia de Datos"),
    ], id="sidebar-header"),

    html.Div([
        html.Div("IDIOMA", className="nav-section-title"),
        dcc.Dropdown(
            id="global-lang",
            options=LANG_OPTIONS,
            value=DEFAULT_LANG,
            clearable=False,
            style={"fontSize": "0.86rem", "marginBottom": "14px"},
        ),
    ]),

    # ── Navegación ────────────────────────────────────────────────────────────
    html.Div([
        html.Div("VISTAS DE NEGOCIO", className="nav-section-title"),
        *[
            html.A([
                html.Span(icon, className="nav-icon"),
                html.Span([
                    html.Strong(label, style={"display": "block", "lineHeight": "1.2"}),
                    html.Span(sublabel, style={"fontSize": "0.72rem", "color": "#8080a0"}),
                ]),
            ],
            href=href,
            id=f"nav-{href.strip('/')}",
            className="nav-link-custom",
            )
            for href, icon, label, sublabel in NAV_ITEMS
        ],
    ], id="nav-menu"),

    # ── Footer ────────────────────────────────────────────────────────────────
    html.Div([
        "Dash + Plotly · Python",
        html.Br(),
        "SCY1101 · 2026",
    ], id="sidebar-footer"),
], id="sidebar")

# ── Layout principal ──────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    sidebar,
    html.Div(
        dcc.Loading(
            id="loading-page",
            type="circle",
            color="#3b82f6",
            children=html.Div(id="page-content"),
        ),
        id="page-content-wrapper",
    ),
], id="wrapper")


# ── Callback de routing ───────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname in (None, "/", ""):
        # Redirigir a vista ejecutiva por defecto
        return page_ejecutiva.layout()
    pathname = pathname.lower().strip("/")
    pages = {
        "ejecutiva":      page_ejecutiva.layout,
        "operativa":      page_operativa.layout,
        "tecnica":        page_tecnica.layout,
        "overview":       page_overview.layout,
        "eda":            page_eda.layout,
        "transformacion": page_transformacion.layout,
        "comparacion":    page_comparacion.layout,
        "modelado":       page_modelado.layout,
        "evaluacion":     page_evaluacion.layout,
        "optimizacion":   page_optimizacion.layout,
    }
    fn = pages.get(pathname)
    if fn:
        return fn()
    return html.Div([
        html.H2("404 – Página no encontrada"),
        html.P(f"La ruta '{pathname}' no existe."),
        html.A("Volver al inicio", href="/ejecutiva"),
    ], style={"padding": "40px"})


# ── Callback para resaltar enlace activo en el sidebar ───────────────────────
@app.callback(
    *[Output(f"nav-{href.strip('/')}", "className") for href, *_ in NAV_ITEMS],
    Input("url", "pathname"),
)
def set_active_nav(pathname):
    if pathname is None:
        pathname = "/ejecutiva"
    classes = []
    for href, *_ in NAV_ITEMS:
        is_active = (pathname.lower().rstrip("/") == href.lower().rstrip("/")
                     or (pathname in ("/", "") and href == "/ejecutiva"))
        classes.append("nav-link-custom active-nav" if is_active else "nav-link-custom")
    return classes


