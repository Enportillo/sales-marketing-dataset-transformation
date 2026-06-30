"""
Módulo de validación de esquemas y manejo de errores para el pipeline ETL.
Define reglas de validación, tipos esperados, rangos y funciones de logging.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# Configuración de logging
# ────────────────────────────────────────────────────────────────────────────

def setup_etl_logger(log_file: str = "etl_pipeline.log") -> logging.Logger:
    """
    Configura un logger profesional para el pipeline ETL.
    
    Args:
        log_file: Ruta del archivo de log
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger("ETL_Pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Handler para archivo
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Handler para consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ────────────────────────────────────────────────────────────────────────────
# Definición de esquemas esperados
# ────────────────────────────────────────────────────────────────────────────

# Esquema esperado para dataset crudo - columnas mínimas requeridas
SCHEMA_RAW = {
    'age': {'type': 'numeric', 'range': (0, 120), 'nullable': True},
    'gender': {'type': 'category', 'nullable': True},
    'country': {'type': 'category', 'nullable': True},
    'subscription_type': {'type': 'category', 'nullable': False},
    'acquisition_channel': {'type': 'category', 'nullable': False},
    'total_spent': {'type': 'numeric', 'range': (0, 1000000), 'nullable': True},
    'avg_order_value': {'type': 'numeric', 'range': (0, 100000), 'nullable': False},
    'lifetime_value': {'type': 'numeric', 'range': (0, 1000000), 'nullable': False},
    'total_visits': {'type': 'numeric', 'range': (0, 100000), 'nullable': False},
    'avg_session_time': {'type': 'numeric', 'range': (0, 10000), 'nullable': False},
    'pages_per_session': {'type': 'numeric', 'range': (0, 1000), 'nullable': False},
    'support_tickets': {'type': 'numeric', 'range': (0, 10000), 'nullable': False},
    'payment_method': {'type': 'category', 'nullable': False},
    'delivery_delay_days': {'type': 'numeric', 'range': (-100, 1000), 'nullable': False},
    'satisfaction_score': {'type': 'numeric', 'range': (0, 5), 'nullable': True},
}

# Esquema esperado para dataset limpio - columnas mínimas procesadas
SCHEMA_CLEAN = {
    'age': {'type': 'numeric', 'range': (0, 120), 'nullable': False},
    'gender': {'type': 'category', 'nullable': False},
    'country': {'type': 'category', 'nullable': False},
    'subscription_type': {'type': 'category', 'nullable': False},
    'acquisition_channel': {'type': 'category', 'nullable': False},
    'total_spent': {'type': 'numeric', 'range': (0, 1000000), 'nullable': False},
    'avg_order_value': {'type': 'numeric', 'range': (0, 100000), 'nullable': False},
    'lifetime_value': {'type': 'numeric', 'range': (0, 1000000), 'nullable': False},
    'total_visits': {'type': 'numeric', 'range': (0, 100000), 'nullable': False},
    'avg_session_time': {'type': 'numeric', 'range': (0, 10000), 'nullable': False},
    'pages_per_session': {'type': 'numeric', 'range': (0, 1000), 'nullable': False},
    'support_tickets': {'type': 'numeric', 'range': (0, 10000), 'nullable': False},
    'payment_method': {'type': 'category', 'nullable': False},
    'delivery_delay_days': {'type': 'numeric', 'range': (-100, 1000), 'nullable': False},
    'satisfaction_score': {'type': 'numeric', 'range': (0, 5), 'nullable': False},
}


# ────────────────────────────────────────────────────────────────────────────
# Funciones de validación
# ────────────────────────────────────────────────────────────────────────────

def validate_schema(
    df: pd.DataFrame,
    schema: Dict,
    stage_name: str = "ETL",
    logger: logging.Logger = None
) -> Tuple[bool, List[str]]:
    """
    Valida que un DataFrame cumpla con el esquema esperado.
    
    Args:
        df: DataFrame a validar
        schema: Diccionario con definición del esquema
        stage_name: Nombre de la etapa del pipeline (para logging)
        logger: Logger para registrar validaciones
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if logger:
        logger.info(f"[{stage_name}] Iniciando validación de esquema...")
    
    # Validar columnas presentes
    expected_cols = set(schema.keys())
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        msg = f"Columnas faltantes: {missing_cols}"
        errors.append(msg)
        if logger:
            logger.error(f"[{stage_name}] ❌ {msg}")
    
    extra_cols = actual_cols - expected_cols
    if extra_cols:
        msg = f"Columnas inesperadas (se ignorarán): {extra_cols}"
        if logger:
            logger.warning(f"[{stage_name}] ⚠️  {msg}")
    
    # Validar tipo y contenido de cada columna
    for col, rules in schema.items():
        if col not in df.columns:
            continue
        
        col_type = rules.get('type')
        nullable = rules.get('nullable', False)
        
        # Validar nulos
        null_count = df[col].isnull().sum()
        if null_count > 0 and not nullable:
            msg = f"Columna '{col}' contiene {null_count} valores nulos (no permitido)"
            errors.append(msg)
            if logger:
                logger.error(f"[{stage_name}] ❌ {msg}")
        
        # Validar rango (numéricos)
        if col_type == 'numeric':
            range_limits = rules.get('range')
            if range_limits:
                min_val, max_val = range_limits
                out_of_range = df[~df[col].isnull()][(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    msg = f"Columna '{col}' tiene {len(out_of_range)} valores fuera del rango [{min_val}, {max_val}]"
                    if logger:
                        logger.warning(f"[{stage_name}] ⚠️  {msg}")
        
        # Validar valores válidos (categorías)
        if col_type == 'category':
            valid_values = rules.get('values')
            if valid_values:
                invalid = df[~df[col].isnull()][~df[col].isin(valid_values)]
                if len(invalid) > 0:
                    unique_invalid = df[col].unique()
                    msg = f"Columna '{col}' contiene valores inválidos: {unique_invalid}"
                    if logger:
                        logger.warning(f"[{stage_name}] ⚠️  {msg}")
    
    is_valid = len(errors) == 0
    
    if is_valid and logger:
        logger.info(f"[{stage_name}] ✅ Validación de esquema exitosa")
    
    return is_valid, errors


def validate_data_quality(
    df: pd.DataFrame,
    stage_name: str = "ETL",
    logger: logging.Logger = None
) -> Dict:
    """
    Evalúa la calidad general del dataset.
    
    Args:
        df: DataFrame a evaluar
        stage_name: Nombre de la etapa
        logger: Logger
        
    Returns:
        Diccionario con métricas de calidad
    """
    metrics = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'null_ratio': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    if logger:
        logger.info(f"[{stage_name}] Métricas de calidad:")
        logger.info(f"  • Filas: {metrics['total_rows']}")
        logger.info(f"  • Columnas: {metrics['total_cols']}")
        logger.info(f"  • Nulls (%): {metrics['null_ratio']:.2f}%")
        logger.info(f"  • Duplicados: {metrics['duplicate_rows']}")
        logger.info(f"  • Memoria (MB): {metrics['memory_usage_mb']:.2f}")
    
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Decorador para manejo de errores
# ────────────────────────────────────────────────────────────────────────────

def handle_etl_errors(logger: logging.Logger = None):
    """
    Decorador para capturar y registrar errores en funciones ETL.
    
    Args:
        logger: Logger para registrar errores
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                msg = f"Error: archivo no encontrado - {str(e)}"
                if logger:
                    logger.error(f"❌ {msg}")
                raise FileNotFoundError(msg) from e
            except pd.errors.ParserError as e:
                msg = f"Error al parsear archivo - {str(e)}"
                if logger:
                    logger.error(f"❌ {msg}")
                raise ValueError(msg) from e
            except KeyError as e:
                msg = f"Error: columna no encontrada - {str(e)}"
                if logger:
                    logger.error(f"❌ {msg}")
                raise KeyError(msg) from e
            except Exception as e:
                msg = f"Error inesperado en {func.__name__}: {str(e)}"
                if logger:
                    logger.error(f"❌ {msg}")
                raise Exception(msg) from e
        return wrapper
    return decorator
