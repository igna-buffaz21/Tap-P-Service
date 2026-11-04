# Tap-P-Service

Proyecto en Python para entrenar y evaluar modelos de aprendizaje automático con datos de ventas almacenados en PostgreSQL.

Incluye dos flujos principales:

- **Regresión de ventas:** estima las unidades vendidas por producto y semana para una sucursal.
- **Clasificación de clientes:** estima si un cliente volverá a comprar en los próximos 30 días y permite identificar a quienes podrían no regresar.

Actualmente se ejecuta desde la consola. Cada ejecución consulta los datos, prepara las variables, entrena un árbol de decisión y muestra la evaluación y ejemplos de predicciones. Los modelos se devuelven en memoria; los scripts no los guardan en disco ni exponen una API.

## Tecnologías

| Tecnología | Uso |
| --- | --- |
| Python | Lenguaje del proyecto. El entorno local utilizado tiene Python 3.14.4. |
| pandas | Limpieza, agrupación y transformación de datos. |
| NumPy | Operaciones numéricas y construcción de variables de clientes. |
| scikit-learn | Árboles de decisión y métricas de evaluación. |
| psycopg | Conexión y consultas a PostgreSQL. |
| python-dotenv | Carga de variables de entorno desde `.env`. |

Las versiones de las dependencias están fijadas en [requirements.txt](requirements.txt).

## Arquitectura

El código separa la configuración, el acceso a datos, la preparación de variables y el entrenamiento.

```text
app/
├── config.py                     # Variables de entorno y negocio seleccionado
├── database.py                   # Conexiones a las bases TAP1 y TAP2
├── data/
│   ├── sales_repository.py       # Ventas históricas por sucursal y producto
│   └── customer_repository.py    # Compras de clientes del negocio
├── ml/
│   ├── sales_preprocessing.py    # Preparación de series semanales
│   ├── sales_features.py         # Variables históricas de ventas
│   ├── sales_train.py            # Entrenamiento y evaluación de regresión
│   ├── customer_features.py      # Variables históricas y etiquetas de clientes
│   └── customer_train.py         # Entrenamiento y evaluación de clasificación
└── analysis/
    └── rainfall_sales.py         # Análisis exploratorio de lluvia y ventas
```

El recorrido de los datos es:

```text
PostgreSQL → Repositorio → Preparación de datos y variables
           → Separación temporal → Entrenamiento → Evaluación en consola
```

- **Configuración:** `config.py` carga las conexiones y `BUSINESS_ID` desde el entorno.
- **Acceso a datos:** `database.py` abre las conexiones; los repositorios ejecutan consultas SQL parametrizadas y devuelven DataFrames.
- **Preparación:** los módulos de preprocesamiento y variables convierten los registros en ejemplos para los modelos.
- **Entrenamiento:** los módulos `sales_train` y `customer_train` coordinan el flujo y comparan el modelo con una referencia sencilla.

## Instalación

### 1. Requisitos previos

- Python con soporte para las versiones fijadas en `requirements.txt`. Como referencia, el entorno local utiliza Python 3.14.4.
- `pip` y el módulo `venv`.
- Acceso a las bases PostgreSQL con los datos y permisos de lectura sobre las tablas consultadas.

Los comandos siguientes se ejecutan desde la carpeta raíz del repositorio.

### 2. Crear y activar el entorno virtual

En Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows, desde PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configurar el entorno

Crear un archivo `.env` en la raíz del proyecto con las credenciales correspondientes:

```dotenv
DATABASE_URL_TAP1=postgresql://usuario:password@localhost:5432/base_tap1
DATABASE_URL_TAP2=postgresql://usuario:password@localhost:5432/base_tap2
BUSINESS_ID=1
```

| Variable | Descripción |
| --- | --- |
| `DATABASE_URL_TAP1` | Conexión usada para las ventas históricas. |
| `DATABASE_URL_TAP2` | Conexión usada para las compras de clientes. |
| `BUSINESS_ID` | Identificador del negocio a analizar. Si se omite, se utiliza `1`. |

Ambas URLs son obligatorias al importar la configuración, incluso cuando se ejecuta un solo flujo. Reemplazar los valores de ejemplo y mantener las credenciales fuera del control de versiones.

### 5. Contar con los datos necesarios

Los scripts consultan bases existentes; no crean las tablas ni cargan los datos automáticamente.

| Base | Tabla | Columnas consultadas |
| --- | --- | --- |
| TAP1 | `historical_sale` | `branch_id`, `product_id`, `sale_date`, `units_sold` |
| TAP1 | `branch` | `id`, `business_id` |
| TAP2 | `sale` | `customer_id`, `branch_id`, `created_at`, `total_amount`, `business_id`, `status` |

El repositorio de ventas verifica que la sucursal pertenezca al negocio configurado. El repositorio de clientes toma únicamente ventas con estado `COMPLETED` y cliente identificado, convirtiendo la fecha de compra a la zona horaria `America/Argentina/Buenos_Aires`.

## Ejecutar la regresión de ventas

```bash
python -m app.ml.sales_train
```

El programa solicita la sucursal:

```text
Ingresa la sucursal a analizar (44, 45, 47, 3, 46): 44
```

Las opciones del mensaje son orientativas: se acepta cualquier número entero, siempre que la sucursal exista y pertenezca al negocio. Si la entrada no es un entero, se vuelve a solicitar.

### Preparación y modelo

1. Agrupa las ventas por sucursal, producto y semana.
2. Conserva las semanas entre el **07/01/2013 y el 07/08/2017**, intervalo configurado para el historial de Corporación Favorita.
3. Selecciona productos con al menos **40 semanas registradas** en el período de entrenamiento.
4. Completa semanas faltantes con cero y elimina las anteriores a la primera venta registrada de cada producto.
5. Construye las variables: identificador del producto, ventas de las cuatro semanas anteriores, promedio de esas cuatro semanas y diferencia entre las dos últimas.
6. Divide cronológicamente el calendario: 80 % inicial para entrenamiento y 20 % final para prueba.
7. Entrena un `DecisionTreeRegressor` con profundidad máxima de 8, mínimo de 20 muestras por hoja y semilla 42.

### Resultados

La consola muestra reglas del árbol, cantidad de ejemplos, profundidad, hojas, primeras diez predicciones y **MAE** (error absoluto medio).

El MAE expresa el error promedio en unidades vendidas: cuanto menor, mejor. Se compara con un modelo base que predice las mismas ventas de la semana anterior.

La evaluación utiliza ventas históricas reales de semanas anteriores para construir las variables de cada semana de prueba. No implementa una proyección recursiva de varias semanas futuras.

## Ejecutar la clasificación de clientes

```bash
python -m app.ml.customer_train
```

Este flujo utiliza `BUSINESS_ID` y analiza las compras del negocio completo, sin solicitar una sucursal.

### Preparación y modelo

Genera fechas de evaluación cada 30 días, dejando inicialmente 90 días de historial y un período futuro de 30 días para determinar si el cliente regresó.

| Variable | Significado |
| --- | --- |
| `days_since_last_purchase` | Días desde la última compra. |
| `purchases_last_30d` | Cantidad de compras en los últimos 30 días. |
| `purchases_last_90d` | Cantidad de compras en los últimos 90 días. |
| `avg_days_between_purchases` | Intervalo promedio entre compras de todo el historial anterior a la evaluación. |
| `total_spent_last_90d` | Importe total gastado en los últimos 90 días. |

La etiqueta `will_return_30d` vale `1` si el cliente compra en los próximos 30 días y `0` si no compra.

Se separan las fechas de evaluación cronológicamente con una proporción del 80 % para el tramo inicial y 20 % para prueba. Solo se conservan ejemplos de entrenamiento cuyo horizonte de 30 días ya terminó al inicio de la prueba.

Se entrena un `DecisionTreeClassifier` con profundidad máxima de 5, mínimo de 20 muestras por hoja y semilla 42. Si la probabilidad estimada de **no regresar es mayor o igual al 40 %**, se predice `0`; de lo contrario, se predice `1`.

### Resultados y métricas

La consola muestra el tamaño de los conjuntos, datos del árbol, métricas, matriz de confusión y primeras diez predicciones con la probabilidad de no regresar.

**Precisión, recall y F1 toman como clase de interés `0 = no regresa`.**

| Métrica | Interpretación |
| --- | --- |
| Accuracy | Proporción total de predicciones correctas. |
| Precisión | De los clientes marcados como «no regresa», cuántos realmente no regresaron. |
| Recall | De los clientes que realmente no regresaron, cuántos detectó el modelo. |
| F1 | Media armónica entre precisión y recall. |

En la matriz de confusión, las filas son los valores reales y las columnas las predicciones, en orden `[0 = no regresa, 1 = regresa]`. También se muestra una referencia que siempre predice que el cliente regresa.

## Parámetros para ajustar

| Archivo | Configuración |
| --- | --- |
| `.env` | Conexiones y negocio. |
| `app/ml/sales_preprocessing.py` | Fechas del calendario, proporción de entrenamiento y mínimo de semanas por producto. |
| `app/ml/sales_train.py` | Variables de entrada y parámetros del árbol de regresión. |
| `app/ml/customer_features.py` | Horizonte futuro e historial inicial; por defecto, 30 y 90 días. Las variables de compras y gasto mantienen ventanas de 30/90 días definidas en el código. |
| `app/ml/customer_train.py` | Variables de entrada, separación temporal, árbol y umbral de clasificación. |

Para usar ventas de otro período, ajustar `FIRST_WEEK` y `LAST_WEEK` en `sales_preprocessing.py`. Si se cambia el horizonte de clientes, revisar también la separación temporal y las etiquetas de 30 días en `customer_train.py`.

Actualmente, el parámetro `product_limit` de `get_weekly_sales` no limita los productos de la consulta SQL.

## Análisis exploratorio de lluvia y ventas

`app/analysis/rainfall_sales.py` contiene un análisis de correlación entre precipitaciones y ventas, separado de los entrenamientos. Consulta también `historical_weather` y `branch.weather_location_id`.

Actualmente importa `get_connection`, función que no está definida en `app/database.py`. Antes de ejecutarlo se debe actualizar esa referencia a la conexión correspondiente y contar con las tablas meteorológicas. Los entrenamientos principales no dependen de este módulo.

## Problemas frecuentes

- **`KeyError: DATABASE_URL_TAP1` o `DATABASE_URL_TAP2`:** comprobar que ambas variables estén definidas en `.env` o en el entorno.
- **Error de conexión a PostgreSQL:** revisar host, puerto, nombre de base, credenciales y acceso de red.
- **La sucursal no pertenece al negocio:** revisar el número ingresado y `BUSINESS_ID`.
- **No hay datos o ningún producto cumple el criterio de actividad:** comprobar el historial de la sucursal, el intervalo de fechas y el mínimo de semanas requerido.
- **No hay suficientes datos para evaluar o generar ejemplos de clientes:** se necesita historial suficiente para las ventanas de preparación, el horizonte futuro y la separación temporal.
- **Se necesitan ejemplos de ambas clases:** el conjunto de entrenamiento debe incluir clientes que regresan y clientes que no regresan.
- **Error al importar `app`:** ejecutar los comandos con `python -m` desde la raíz del proyecto y con el entorno virtual activado.
