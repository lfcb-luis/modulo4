# Predicción de Resultados en Exámenes de Matemáticas

Este proyecto implementa modelos de Machine Learning para predecir si un estudiante aprobará un examen de matemáticas, utilizando árboles de decisión y redes neuronales con TensorFlow/Keras.

## Información del dataset

| Característica                         | Tipo   |
|----------------------------------|--------|
| is_male                         | bool   |
| ethnicity                       | string |
| parental_level_of_education     | int8   |
| has_standard_lunch              | bool   |
| has_completed_preparation_test  | bool   |
| reading_score                   | int64  |
| writing_score                   | int64  |
| math_score                      | int64  |

## Requisitos

- Python 3.9+
- Docker (para ejecutar en contenedor)
- [uv](https://github.com/astral-sh/uv) para la gestión de paquetes y entornos viruales

### Ejecución en equipo locales

1. **Inicializar el entorno del proyecto:**

   ```bash
   uv init
   ```

2. **Activación del entorno:**

   ```bash
   .venv/Scripts/activate
   ```

3. **Instalar dependencias dentro del entorno virtual:**

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Iniciar MLflow en el puerto 5000:**

Iniciar el servidor local de MLflow, ejecutando:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

5. **Instalar dependencias dentro del entorno virtual:**

```bash
python app.py
```
6. Visualizar los experimentos en [http://127.0.0.1:5000](http://127.0.0.1:5000).

![Metricas](<Imagen de WhatsApp 2025-03-03 a las 00.19.22_2ea30e90.jpg>)

Esto entrenará los modelos y registrará los resultados en MLflow.

## Ejecución en Docker

Para ejecutar este proyecto dentro de un contenedor Docker:

1. **Construir la imagen:**

   ```bash
   docker build -t aprobacionmate:v1 .
   ```

2. **Ejecutar el contenedor:**

   ```bash
   docker run -d  -p 5000:5000 aprobacionmate:v1   
   ```

Esto iniciará el entrenamiento de los modelos dentro del contenedor y los resultados se registrarán en MLflow.

3. Visualizar los experimentos en [http://localhost:5000]([http://localhost:5000).

## Contribución

- Gabriel Antonio Vallejo Loaiza -  2250145
- Juan Fernando Rodriguez - 2240585
- Luis Felipe Carabali Balanta - 2244790
- Gregth Raynell Hernández Buenaño - 2250194
