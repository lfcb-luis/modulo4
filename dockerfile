FROM python:3.9


WORKDIR /app


COPY . /app/

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 5000

# Inicio del servidor de Mlflow y de la aplicación
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 & \
    python app.py
