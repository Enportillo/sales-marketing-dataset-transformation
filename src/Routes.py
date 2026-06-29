from pathlib import Path

DIRECTORIO_PRINCIPAL = Path(__file__).parent.parent

RUTAS = {
    'data' : DIRECTORIO_PRINCIPAL / 'data',
    'notebooks' : DIRECTORIO_PRINCIPAL / 'notebooks',
    'data_processed' : DIRECTORIO_PRINCIPAL / 'data' / 'processed',
    'data_raw' : DIRECTORIO_PRINCIPAL / 'data' / 'raw',
    'figures' : DIRECTORIO_PRINCIPAL / 'outputs' / 'figures',
    'imagenes' : DIRECTORIO_PRINCIPAL / 'outputs' / 'figures',
    'models'   : DIRECTORIO_PRINCIPAL / 'outputs' / 'models'
}

