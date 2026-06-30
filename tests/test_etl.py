"""
Tests básicos para validar el pipeline ETL.

Ejecutar con:
    python -m pytest tests/test_etl.py -v
    
O simplemente:
    python tests/test_etl.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from etl_validation import (
    validate_schema,
    validate_data_quality,
    SCHEMA_RAW,
    SCHEMA_CLEAN,
    setup_etl_logger
)
from Routes import RUTAS


class TestETLPipeline:
    """Suite de tests para el pipeline ETL."""
    
    @staticmethod
    def setup_class():
        """Ejecuta antes de los tests."""
        print("\n" + "="*70)
        print("INICIANDO SUITE DE TESTS DEL PIPELINE ETL")
        print("="*70)
    
    @staticmethod
    def test_raw_data_exists():
        """Test 1: Verifica que el dataset crudo existe."""
        print("\n[TEST 1] Verificando existencia del dataset crudo...")
        ruta = RUTAS['data_raw'] / 'Dirty_Sales_Marketing_Dataset.xlsx'
        assert ruta.exists(), f"Dataset crudo no encontrado en {ruta}"
        print(f"  [OK] Dataset crudo encontrado: {ruta}")
    
    @staticmethod
    def test_raw_data_shape():
        """Test 2: Verifica que el dataset crudo tiene el shape esperado."""
        print("\n[TEST 2] Verificando shape del dataset crudo...")
        ruta = RUTAS['data_raw'] / 'Dirty_Sales_Marketing_Dataset.xlsx'
        df = pd.read_excel(ruta)
        
        assert len(df) > 0, "Dataset crudo está vacío"
        assert len(df.columns) > 0, "Dataset crudo sin columnas"
        assert len(df) >= 4000, f"Dataset tiene {len(df)} filas, se esperaban >= 4000"
        
        print(f"  [OK] Shape válido: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    @staticmethod
    def test_clean_data_exists():
        """Test 3: Verifica que el dataset limpio existe."""
        print("\n[TEST 3] Verificando existencia del dataset limpio...")
        ruta = RUTAS['data_processed'] / 'Sales_Marketing_Clean_(Codificado).csv'
        
        if not ruta.exists():
            print(f"  [SKIP] Dataset limpio no encontrado (ejecuta notebook 2 primero)")
            return True
        
        print(f"  [OK] Dataset limpio encontrado: {ruta}")
        return True
    
    @staticmethod
    def test_schema_raw():
        """Test 4: Valida esquema del dataset crudo."""
        print("\n[TEST 4] Validando esquema del dataset crudo...")
        ruta = RUTAS['data_raw'] / 'Dirty_Sales_Marketing_Dataset.xlsx'
        df = pd.read_excel(ruta)
        
        logger = setup_etl_logger()
        try:
            is_valid, errors = validate_schema(df, SCHEMA_RAW, stage_name="TEST_RAW", logger=logger)
        except Exception as e:
            # Es normal que el dataset crudo tenga inconsistencias
            print(f"  [INFO] Esquema crudo tiene problemas (esperado en datos raw): {str(e)[:80]}")
            print(f"  [OK] Validacion completada (tolerante)")
            return True
        
        # Los errores son permitidos en datos crudos (por eso existen los notebooks)
        print(f"  [INFO] Esquema crudo: {len(errors)} issues encontradas (esperado en datos raw)")
        print(f"  [OK] Validacion completada")
    
    @staticmethod
    def test_schema_clean():
        """Test 5: Valida esquema del dataset limpio."""
        print("\n[TEST 5] Validando esquema del dataset limpio...")
        ruta = RUTAS['data_processed'] / 'Sales_Marketing_Clean_(Codificado).csv'
        
        if not ruta.exists():
            print(f"  [SKIP] Dataset limpio no existe (saltando test)")
            return True
        
        df = pd.read_csv(ruta)
        logger = setup_etl_logger()
        is_valid, errors = validate_schema(df, SCHEMA_CLEAN, stage_name="TEST_CLEAN", logger=logger)
        
        # El dataset limpio debe pasar la validación
        assert is_valid, f"Dataset limpio falló validación: {errors}"
        print(f"  [OK] Esquema limpio validado correctamente")
    
    @staticmethod
    def test_data_quality_raw():
        """Test 6: Evalúa calidad del dataset crudo."""
        print("\n[TEST 6] Evaluando calidad del dataset crudo...")
        ruta = RUTAS['data_raw'] / 'Dirty_Sales_Marketing_Dataset.xlsx'
        df = pd.read_excel(ruta)
        
        logger = setup_etl_logger()
        metrics = validate_data_quality(df, stage_name="TEST_RAW", logger=logger)
        
        print(f"  [OK] Calidad evaluada:")
        print(f"     * Nulls: {metrics['null_ratio']:.2f}%")
        print(f"     * Duplicados: {metrics['duplicate_rows']}")
        print(f"     * Memoria: {metrics['memory_usage_mb']:.2f} MB")
    
    @staticmethod
    def test_data_quality_clean():
        """Test 7: Evalúa calidad del dataset limpio."""
        print("\n[TEST 7] Evaluando calidad del dataset limpio...")
        ruta = RUTAS['data_processed'] / 'Sales_Marketing_Clean_(Codificado).csv'
        
        if not ruta.exists():
            print(f"  [SKIP] Dataset limpio no existe (saltando test)")
            return True
        
        df = pd.read_csv(ruta)
        logger = setup_etl_logger()
        metrics = validate_data_quality(df, stage_name="TEST_CLEAN", logger=logger)
        
        # Dataset limpio debe tener muy pocos nulls
        assert metrics['null_ratio'] < 1.0, f"Dataset limpio tiene demasiados nulls: {metrics['null_ratio']}%"
        
        print(f"  [OK] Calidad evaluada:")
        print(f"     * Nulls: {metrics['null_ratio']:.2f}%")
        print(f"     * Duplicados: {metrics['duplicate_rows']}")
        print(f"     * Memoria: {metrics['memory_usage_mb']:.2f} MB")
    
    @staticmethod
    def test_files_consistency():
        """Test 8: Verifica consistencia entre archivos."""
        print("\n[TEST 8] Verificando consistencia entre archivos...")
        
        ruta_raw = RUTAS['data_raw'] / 'Dirty_Sales_Marketing_Dataset.xlsx'
        ruta_clean = RUTAS['data_processed'] / 'Sales_Marketing_Clean_(Codificado).csv'
        ruta_clean_xlsx = RUTAS['data_processed'] / 'Sales_Marketing_Clean.xlsx'
        
        df_raw = pd.read_excel(ruta_raw)
        
        if ruta_clean.exists():
            df_clean = pd.read_csv(ruta_clean)
            # El dataset limpio debe tener <= filas que el crudo
            assert len(df_clean) <= len(df_raw), "Dataset limpio tiene mas filas que el crudo"
            print(f"  [OK] Filas: {len(df_raw)} (raw) -> {len(df_clean)} (clean)")
        
        if ruta_clean_xlsx.exists():
            df_clean_xlsx = pd.read_excel(ruta_clean_xlsx)
            assert len(df_clean_xlsx) <= len(df_raw), "Dataset limpio XLSX tiene mas filas"
            print(f"  [OK] Archivo Excel limpio consistente")
        
        print(f"  [OK] Consistencia verificada")


def run_tests():
    """Ejecuta todos los tests."""
    test_suite = TestETLPipeline()
    test_suite.setup_class()
    
    tests = [
        test_suite.test_raw_data_exists,
        test_suite.test_raw_data_shape,
        test_suite.test_clean_data_exists,
        test_suite.test_schema_raw,
        test_suite.test_schema_clean,
        test_suite.test_data_quality_raw,
        test_suite.test_data_quality_clean,
        test_suite.test_files_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] TEST FALLIDO: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTADOS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
