"""
Punto de entrada principal del proyecto.
Lanza el dashboard interactivo de Plotly Dash.

Uso:
    python main.py
    # Abre http://127.0.0.1:8050 en el navegador
"""

from dashboard.app import app

if __name__ == "__main__":
    print("=" * 60)
    print(" Sales & Marketing Dashboard – Iniciando...")
    print(" URL: http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=True, use_reloader=False, port=8050, host="0.0.0.0")
