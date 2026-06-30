"""
Punto de entrada principal del proyecto.
Lanza el dashboard interactivo de Plotly Dash.

Uso:
    python main.py
    # Abre http://127.0.0.1:8050 en el navegador
"""

import os
from dashboard.app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    host = os.getenv("HOST", "127.0.0.1")
    print("=" * 60)
    print(" Sales & Marketing Dashboard – Iniciando...")
    if host == "0.0.0.0":
        # 0.0.0.0 is a bind address, not a browser URL.
        print(f" URL local: http://127.0.0.1:{port}")
    else:
        print(f" URL: http://{host}:{port}")
    print("=" * 60)
    app.run(debug=True, use_reloader=False, port=port, host=host)
