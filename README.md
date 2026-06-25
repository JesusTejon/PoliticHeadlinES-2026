# PoliticHeadlinES - I2C PERSEUS UHU

Este repositorio contiene el código fuente y el cuaderno de experimentación utilizado para la tarea de selección de titulares de noticias políticas. El sistema evalúa la relevancia semántica entre el cuerpo de una noticia y diferentes candidatos a titular, formulando el problema como una tarea de clasificación binaria.

## ⚙️ Descripción del Sistema

El flujo de trabajo (pipeline) implementado en el cuaderno principal sigue las siguientes fases principales:

### 1. Preprocesamiento de Datos y Estrategia *Head+Tail*
Debido a la longitud de los artículos periodísticos, se aplica una técnica de truncamiento asimétrico (Head+Tail). 
* Se recorta el texto para mantener un máximo de 350 palabras.
* El sistema preserva la mitad de ese límite desde el inicio del artículo y la otra mitad desde el final, uniendo ambas partes con un token especial `[SALTO]`.
* Para entrenar el modelo clasificador, el dataset original (que contiene 10 columnas de títulos) se transforma al formato *melt* (una fila por cada par noticia-titular).
* Se aplica un balanceo de clases estricto, manteniendo el titular correcto (etiqueta 1) y muestreando aleatoriamente solo 2 titulares falsos (etiqueta 0) por cada noticia.

### 2. Búsqueda de Hiperparámetros (Optuna)
El cuaderno integra el framework `optuna` para buscar la mejor configuración de hiperparámetros de manera automatizada.
* Se ejecutan 10 *trials* (pruebas) buscando optimizar el `eval_f1` (F1-score en validación).
* El espacio de búsqueda explora la tasa de aprendizaje (learning rate), decaimiento de pesos (weight decay), ratio de calentamiento (warmup ratio) y el número de épocas de entrenamiento.

### 3. Evaluación (PA-nDCG)
Aunque el modelo se entrena maximizando el F1-Score, el cuaderno incluye funciones completas para reordenar las probabilidades generadas por el modelo y evaluar el rendimiento utilizando la métrica oficial de la competición: **PA-nDCG**.

---

## 🛠️ Requisitos e Instalación

El código está diseñado para ejecutarse en un entorno con aceleración GPU (como Google Colab con NVIDIA T4)[cite: 4]. Las principales dependencias son[cite: 4]:

* `torch`
* `transformers`
* `datasets`
* `optuna`
* `scikit-learn`
* `pandas` y `numpy`

Para instalar las dependencias necesarias de optimización, puedes ejecutar:
```bash
pip install optuna
