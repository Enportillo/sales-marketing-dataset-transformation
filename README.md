# Sales Marketing Dataset

Proyecto de analisis, transformacion y comparacion de resultados para un dataset de ventas y marketing.

## Objetivo

Estandarizar un dataset con problemas de calidad (nulos, formatos inconsistentes y valores extremos), generar versiones limpias para analisis/modelado y comparar resultados antes vs despues de la transformacion.

## Preparacion del entorno

1. Crear entorno virtual:

```bash
python -m venv venv
```

2. Activar entorno virtual:

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Estructura de trabajo

- notebooks/1_EDA.ipynb: analisis exploratorio inicial.
- notebooks/2_Transformacion_de_datos.ipynb: limpieza, imputacion, winsorizacion, codificacion y exportaciones.
- notebooks/3_Comparacion_de_resultados.ipynb: comparacion entre dataset sucio y limpio + generacion de figuras.
- notebooks/4_supervised_modeling.ipynb: segmentacion (clustering) + simulacion inicial de campana de upselling.
- notebooks/5_model_evaluation.ipynb: evaluacion comparativa de modelos de clasificacion y regresion con narrativa de conclusiones.
- notebooks/6_hyperparameter_optimization.ipynb: optimizacion de hiperparametros (v1-v4), analisis de matrices y ajuste de umbral operativo.
- src/Routes.py: rutas centralizadas del proyecto (raw, processed, notebooks, figures).

## Orden recomendado de ejecucion

1. Ejecutar notebooks/1_EDA.ipynb.
2. Ejecutar notebooks/2_Transformacion_de_datos.ipynb para generar datasets de salida en data/processed/.
3. Ejecutar notebooks/3_Comparacion_de_resultados.ipynb para generar comparativas y figuras en outputs/figures/.
4. Ejecutar notebooks/4_supervised_modeling.ipynb para validar segmentacion de clientes y simulacion inicial de upselling.
5. Ejecutar notebooks/5_model_evaluation.ipynb para comparar desempeno de Random Forest vs Regresion Logistica y evaluar regresion de total_spent.
6. Ejecutar notebooks/6_hyperparameter_optimization.ipynb para optimizar versiones v1-v4, justificar decisiones de modelo y calibrar umbral de campana.

## Parte 4 - Modelado y simulacion inicial (notebook 4)

En esta parte se combinan dos bloques:

1. No supervisado (clustering):
	 - Variables de comportamiento seleccionadas para segmentacion operativa.
	 - Escalado con StandardScaler para evitar sesgo por magnitud.
	 - Validacion con Elbow + Silhouette y eleccion de K=3 por accionabilidad de negocio.
2. Supervisado (simulacion de upselling):
	 - Entrenamiento base de Random Forest y Regresion Logistica.
	 - Simulacion con umbral alto (0.75) sobre clientes basicos.
	 - Resultado observado: Random Forest identifica 1 cliente potencial; Regresion Logistica identifica 0.

Decision metodologica:

- Se prioriza Random Forest para la siguiente fase porque muestra mayor capacidad para capturar patrones no lineales del problema de conversion.

## Parte 5 - Evaluacion de modelos (notebook 5)

Esta parte consolida la evaluacion tecnica y narrativa:

- Clasificacion:
	- Comparacion de Random Forest vs Regresion Logistica con reportes y matrices.
	- Conclusiones estandarizadas en formato informe: diagnostico, interpretacion, cierre y siguiente accion.
- Regresion (target: total_spent):
	- Comparacion de LinearRegression vs RandomForestRegressor.
	- Metricas usadas: RMSE y R2.
	- Resultado observado en la corrida actual: RandomForestRegressor supera a LinearRegression (menor RMSE y mayor R2).

Decision metodologica:

- Se mantiene Random Forest como baseline principal para clasificacion y regresion en este dataset, usando modelos lineales como benchmark de control.

## Parte 6 - Optimizacion de hiperparametros (notebook 6)

Esta parte implementa una ruta iterativa de mejora v1-v4 enfocada en upselling:

- v1: baseline con grilla amplia.
- v2: mismas tecnicas con solo columnas de comportamiento.
- v3: objetivo orientado a negocio (clase positiva = upgrade).
- v4: RandomizedSearchCV optimizado por Average Precision.

Metricas destacadas de la ultima corrida:

- v1 ROC-AUC test: 0.6094
- v2 ROC-AUC test: 0.6059
- v3 ROC-AUC test: 0.6146
- v4 ROC-AUC test: 0.6322
- v4 Average Precision test: 0.6594

Calibracion de decision:

- Se calcula umbral operativo por maximo F1.
- Umbral recomendado observado: 0.3378
- Precision/Recall en ese punto: 0.5507 / 0.9838
- Interpretacion: alta captura de oportunidades (recall alto) con costo de mas falsos positivos; el umbral debe ajustarse segun presupuesto y capacidad comercial.

## Transformaciones aplicadas

### Imputacion de datos

Reglas implementadas en notebooks/2_Transformacion_de_datos.ipynb:

- age:
	- Limpieza previa de strings (espacios y variantes de nan como texto).
	- Imputacion por mediana global.
	- Conversion final a entero.
- total_spent:
	- Imputacion por mediana dentro de cada subscription_type.
	- Fallback a mediana global si algun grupo queda sin valor imputable.
- satisfaction_score:
	- Imputacion por mediana global.
- gender:
	- Estandarizacion de formato y reemplazo de nulos por Unknown.
- country:
	- Imputacion por moda global.

### Tratamiento de outliers

Se aplica winsorizacion basada en IQR (sin eliminar filas):

- Regla: limites en Q1 - 1.5 * IQR y Q3 + 1.5 * IQR.
- Accion: clip de valores fuera de limites.
- Variables tratadas:
	- age, total_spent, avg_order_value, lifetime_value, total_visits,
		avg_session_time, pages_per_session, support_tickets y delivery_delay_days.
- Post-proceso:
	- Restauracion de tipo entero en age, total_visits, support_tickets y delivery_delay_days.

### Codificacion con LabelEncoder

Despues de construir df_limpio, se crea un segundo dataset (df_codificado) usando LabelEncoder de scikit-learn.

Columnas codificadas:

- gender
- country
- acquisition_channel
- subscription_type
- payment_method

Nota importante:

- La codificacion se ajusta en el mismo notebook por columna y transforma solo el dataset final codificado.
- El dataset limpio original (sin codificar) se conserva por separado para analisis interpretables.

## Guardado diferenciado de archivos resultantes

El pipeline genera salidas separadas segun objetivo de uso:

1. Dataset limpio para analisis (formato tabular original):
	 - data/processed/Sales_Marketing_Clean.xlsx
2. Dataset codificado para modelado:
	 - data/processed/Sales_Marketing_Clean_(Codificado).csv
3. Figuras de comparacion (notebook 3):
	 - outputs/figures/conteo_nulos_bar.png
	 - outputs/figures/heatmap_nulos.png
	 - outputs/figures/conteo_generos_sucios_bar.png
	 - outputs/figures/conteo_generos_limpios_bar.png

## Validaciones tecnicas implementadas

- Optimizacion de memoria con downcasting y medicion cuantitativa del impacto en bytes y porcentaje.
- Agrupacion multivariable por country y subscription_type.
- Tabla dinamica con pivot_table para acquisition_channel vs country.

## Notas operativas

- Las rutas se gestionan desde src/Routes.py mediante el diccionario RUTAS.
- data/processed/ y outputs/figures/ contienen artefactos generados por notebooks.
- El proyecto ignora archivos de documentacion (docs/) y comprimidos (*.zip) mediante .gitignore para evitar versionar adjuntos pesados o de apoyo.
