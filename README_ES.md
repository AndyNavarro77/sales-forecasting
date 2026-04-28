# 📈 Forecasting de Ventas — Análisis de Series de Tiempo con Prophet

> **Proyecto de forecasting de ventas end-to-end usando Facebook Prophet y análisis estadístico de series de tiempo — prediciendo ingresos mensuales 12 meses hacia adelante para un negocio retail con 4 años de datos históricos.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-orange)](https://facebook.github.io/prophet/)
[![Pandas](https://img.shields.io/badge/Pandas-Análisis-green?logo=pandas)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?logo=jupyter)](https://jupyter.org)
[![Estado](https://img.shields.io/badge/Estado-Activo-brightgreen)]()

---

## 🧠 El Problema de Negocio

Todo negocio retail enfrenta el mismo desafío de planificación: ¿cuánto venderemos el próximo mes? ¿El próximo trimestre?

Sin forecasts confiables, las empresas enfrentan dos modos de falla costosos:
- **Subestimar la demanda** → quiebre de stock, ventas perdidas, clientes frustrados
- **Sobrestimar la demanda** → inventario excedente, capital inmovilizado, costos innecesarios

El riesgo es mayor en Q4 — cuando un mal mes puede definir todo el año.

> *Este proyecto construye un sistema de forecasting que predice ventas mensuales 12 meses hacia adelante, cuantifica la incertidumbre con intervalos de confianza y entrega proyecciones trimestrales para la planificación operativa.*

---

## ✅ La Solución

Un pipeline de análisis de series de tiempo end-to-end que descompone las ventas históricas en tendencia, estacionalidad y ruido — valida cada componente estadísticamente — y entrena Facebook Prophet para generar forecasts con outputs listos para el negocio.

> *De 4 años de datos transaccionales crudos a un forecast de 12 meses con desglose trimestral — en un único notebook reproducible.*

---

## 📐 Arquitectura del Sistema

```
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  Dataset Superstore │───▶│  EDA Series Tiempo   │───▶│  Prueba Estadística  │
│  (4 años · 9.994   │    │  Tendencia · Estac.  │    │  Test Mann-Kendall   │
│   transacciones)    │    │  Descomposición      │    │  H0/H1 · p-valor     │
└─────────────────────┘    └──────────────────────┘    └──────────┬───────────┘
                                                                    │
                                              ┌─────────────────────▼──────────┐
                                              │   Modelo de Forecasting Prophet │
                                              │   Forecast 12 meses · IC        │
                                              │   Resumen ejecutivo trimestral  │
                                              └────────────────────────────────┘
```

---

## 🔄 Metodología — Framework STAR

### Situación
Una empresa retail con 4 años de datos de ventas (2014-2017) en 3 categorías de productos y 4 regiones geográficas necesitaba forecastear ingresos para 2018 para soportar decisiones de inventario, personal y presupuesto.

### Tarea
Construir un pipeline de forecasting que:
- Identifique y valide patrones estadísticos en las ventas históricas
- Descomponga la serie temporal en componentes interpretables
- Entrene y evalúe un modelo de forecasting contra un baseline naive
- Entregue una proyección de 12 meses con intervalos de confianza y desglose trimestral

### Acción

**1 — Análisis Exploratorio de Datos**
- Agregó 9.994 transacciones en 48 puntos de datos mensuales (2014-2017)
- Identificó estacionalidad fuerte: Noviembre promedia $80.115 (mejor mes) vs Febrero en $14.938 (peor mes) — diferencia de 5x
- Confirmó tendencia alcista: ventas mensuales promedio crecieron de ~$40.000 a ~$62.000 en 4 años

**2 — Descomposición de la Serie Temporal**
Aplicó descomposición multiplicativa para separar tres componentes:
- **Tendencia:** crecimiento sostenido de $40.518 a $61.650 (+52,2% en 4 años)
- **Estacionalidad:** Q4 consistentemente 75% por encima del promedio; Q1 consistentemente 50% por debajo
- **Residuos:** ruido aleatorio sin patrón detectable — el modelo captura bien la señal

**3 — Prueba de Hipótesis (Test de Mann-Kendall)**
Verificó formalmente si la tendencia observada es estadísticamente significativa o podría deberse a variación aleatoria:

| | Resultado |
|--|--|
| H0 | No existe tendencia en las ventas mensuales |
| H1 | Existe una tendencia estadísticamente significativa |
| Kendall's Tau | 0,378 |
| P-valor | 0,000153 |
| **Conclusión** | **Rechazar H0 — tendencia estadísticamente significativa (p < 0,001)** |

El crecimiento del 52,2% en 4 años no es aleatorio — refleja expansión genuina del negocio, justificando la inversión en infraestructura de forecasting.

**4 — Entrenamiento y Evaluación del Modelo Prophet**

| Métrica | Prophet | Baseline Naive |
|---------|---------|---------------|
| MAPE | 28,6% | 24,5% |
| MAE | $14.412 | — |
| RMSE | $17.647 | — |

El baseline naive (repetir los valores del año anterior) es competitivo — resultado común con solo 4 años de datos y alta volatilidad mensual. Este es un hallazgo honesto: Prophet agrega valor en la proyección de tendencia y cuantificación de incertidumbre, incluso cuando la precisión puntual es comparable a métodos más simples.

**5 — Forecast 2018 y Output de Negocio**

| Trimestre | Ventas Proyectadas | % del Anual |
|-----------|-------------------|-------------|
| Q1 (Ene-Mar) | $126.870 | 16,3% |
| Q2 (Abr-Jun) | $147.523 | 19,0% |
| Q3 (Jul-Sep) | $213.197 | 27,4% |
| Q4 (Oct-Dic) | $290.743 | 37,4% |
| **Total 2018** | **$778.332** | **+6,2% vs 2017** |

### Resultados

**Resultado Clave #1:** La tendencia alcista de ventas es estadísticamente significativa (p < 0,001) — el negocio está genuinamente creciendo, no experimentando fluctuación aleatoria.

**Resultado Clave #2:** Q4 concentra el 37,4% del revenue anual. Cualquier disrupción operativa entre octubre y diciembre tiene impacto desproporcionado en los resultados del año completo.

**Resultado Clave #3:** El modelo proyecta $778.332 en ventas totales 2018 (+6,2% vs 2017), con Noviembre como mes pico en $121.137 y Febrero como valle en $19.564.

**Resultado Clave #4:** El MAPE de Prophet del 28,6% vs 24,5% del baseline revela un insight importante — con datos históricos limitados y alta volatilidad, incorporar variables externas (promociones, condiciones de mercado) mejoraría significativamente la precisión del forecast.

---

## 📊 Análisis y Visualizaciones

**Serie temporal mensual y patrón de estacionalidad:**

![Time Series Overview](img/time_series_overview.png)

**Descomposición de la serie temporal — tendencia, estacionalidad y residuos:**

![Decomposition](img/decomposition.png)

**Forecast 2018 vs real 2017 — comparación mes a mes:**

![Forecast](img/forecast.png)

**Evaluación del modelo — Prophet vs baseline naive + real vs predicho:**

![Model Evaluation](img/model_evaluation.png)

**Resumen ejecutivo — forecast mensual 2018 y desglose trimestral:**

![Executive Summary](img/executive_summary.png)

---

## 🔍 Insights Clave de Negocio

| Insight | Acción Recomendada |
|---------|-------------------|
| Q4 = 37,4% del revenue anual | Planificar inventario, personal y logística 90 días antes de octubre |
| Febrero es consistentemente el mes más débil ($14.938 promedio) | Usar Q1 para mejoras operativas, capacitación y preparación |
| Crecimiento del 52,2% en 4 años (estadísticamente significativo) | La infraestructura de forecasting está justificada — el negocio genuinamente se expande |
| La tendencia se acelera en 2017 | Revisar el forecast trimestralmente — la tasa de crecimiento puede superar la proyección del 6,2% |
| Alta volatilidad mensual (MAPE 28,6%) | Incorporar calendario promocional y factores externos para mejorar la precisión |

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología | Propósito |
|------|------------|-----------|
| Fuente de Datos | Kaggle — Superstore Sales Dataset | 9.994 transacciones retail en 4 años |
| EDA y Preprocesamiento | Python · Pandas · Matplotlib · Seaborn | Agregación y visualización de series temporales |
| Análisis Estadístico | SciPy · Statsmodels | Descomposición y prueba de hipótesis Mann-Kendall |
| Forecasting | Facebook Prophet | Forecast de 12 meses con intervalos de confianza |
| Evaluación | scikit-learn · MAE · RMSE · MAPE | Precisión del modelo vs baseline naive |
| Entorno | Jupyter Notebook | Pipeline completo y reproducible |

---

## 📁 Estructura del Repositorio

```
sales-forecasting/
│
├── notebooks/
│   └── 01_sales_forecasting.ipynb   # Pipeline completo: EDA, descomposición, prueba de hipótesis, forecast
├── data/
│   └── Sample - Superstore.csv      # Dataset fuente — 4 años de transacciones retail
├── img/
│   ├── time_series_overview.png     # Ventas mensuales + patrón de estacionalidad
│   ├── decomposition.png            # Tendencia · estacionalidad · residuos
│   ├── forecast.png                 # Forecast 2018 + comparación 2017 vs 2018
│   ├── model_evaluation.png         # Prophet vs baseline naive + real vs predicho
│   └── executive_summary.png        # Forecast mensual 2018 + desglose trimestral
├── requirements.txt                 # Dependencias Python
└── LICENSE                          # Licencia MIT
```

---

## 📊 Dataset

Este proyecto usa el dataset **Sample Superstore**, disponible en [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).

El dataset contiene datos de transacciones retail incluyendo fechas de orden, categorías de productos, regiones, ventas, cantidad, descuento y profit — abarcando enero 2014 a diciembre 2017.

---

## 👤 Autor

**Andrés Navarro**
Analista de Datos · Data Science · Python · Series de Tiempo · Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-AndyNavarro77-black?logo=github)](https://github.com/AndyNavarro77)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-blue?logo=linkedin)](https://www.linkedin.com/in/andr%C3%A9s-navarro77/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visitar-orange?logo=netlify)](https://andres-navarro-portfolio.netlify.app/)

---

*Construido para demostrar pensamiento de series de tiempo end-to-end — desde la validación estadística de tendencias hasta forecasts trimestrales accionables — usando herramientas estándar de la industria aplicables a cualquier negocio generador de ingresos.*