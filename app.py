# Importación de librerias

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_training_history(history):
    """Graficar el historial de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Presición
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Perdida
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    
    return fig 

# Carga del dataset de resultados de matemáticas

dataset = load_dataset("mstz/student_performance", "math")
dataset = dataset['train'].to_pandas()

# Definición de las variables de entrada y salida
X=dataset.drop('has_passed_math_exam',axis=1)
y=dataset['has_passed_math_exam']

categoricas = ['ethnicity', 'parental_level_of_education']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Xenco = encoder.fit_transform(X[categoricas])
Xenco_df = pd.DataFrame(Xenco , columns=encoder.get_feature_names_out(categoricas))

# Reemplazo de las variables originales por las codificadas
X = X.drop(categoricas, axis=1)
X = pd.concat([X, Xenco_df], axis=1)

# Separación del dataset para entrenamiento y validación

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición del modelo de DecisionTree para la predicción de la variable y

params = {
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

modelo = DecisionTreeClassifier(**params) #Iniciación del modelo
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test) #Predicción de los datos de prueba

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Ajuste de las variables en booleans para las redes neuronal

X_trainNN = X_train.astype('float32')
X_testNN = X_test.astype('float32')

# Red Neuronal con Dropout
modelNNdrop = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),  # Regularización
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
modelNNdrop.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo con historial
EPOCHSNNdrop = 100
historyNNdrop = modelNNdrop.fit(X_trainNN, y_train, epochs=EPOCHSNNdrop, batch_size=32, validation_data=(X_testNN, y_test))

# Evaluación del modelo
y_predNNdrop = (modelNNdrop.predict(X_testNN) > 0.5).astype("int32")
accuracyNNdrop = accuracy_score(y_test, y_predNNdrop)
precisionNNdrop = precision_score(y_test, y_predNNdrop)
recallNNdrop = recall_score(y_test, y_predNNdrop)
f1NNdrop = f1_score(y_test, y_predNNdrop)


# Definición del modelo de red neuronal para la predicción de la variable y
modeloKeras = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
modeloKeras.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo con historial
EPOCHS = 100
history = modeloKeras.fit(X_trainNN, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_testNN, y_test))

# Evaluación del modelo
y_predNN = (modeloKeras.predict(X_testNN) > 0.5).astype("int32")
accuracyNN = accuracy_score(y_test, y_predNN)
precisionNN = precision_score(y_test, y_predNN)
recallNN = recall_score(y_test, y_predNN)
f1NN = f1_score(y_test, y_predNN)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("Predicción de Resultado Prueba de Matemáticas")

# Start an MLflow run Keras  con Dropout
with mlflow.start_run():

    # Almacenar parametros del modelo
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "binary_crossentropy")
    mlflow.log_param("epochs", EPOCHSNNdrop)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("dropout_rate", 0.3)
    mlflow.log_param("layers", [128, 64, 32, 1])

    # Métricas de precision del modelo
    mlflow.log_metric("accuracy", accuracyNNdrop)
    mlflow.log_metric("precision", precisionNNdrop)
    mlflow.log_metric("recall", recallNNdrop)
    mlflow.log_metric("f1", f1NNdrop)
    
    # Almacenar función de costo del modelo
    
    mlflow.log_figure(plot_training_history(historyNNdrop), "lossNNdrop.png")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("model_type", "Red Neuronal Keras con Dropout")

    # Infer the model signature
    signature = infer_signature(X_trainNN, modelNNdrop.predict(X_trainNN))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=modelNNdrop,
        artifact_path="matematicas-model-Keras-dropout",
        signature=signature,
        input_example=X_trainNN,
        registered_model_name="matematicas-Keras-dropout",
    )

# Start an MLflow run Keras
with mlflow.start_run():

    # Almacenar parametros del modelo
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "binary_crossentropy")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("layers", [64, 32, 1])

    # Métricas de precision del modelo
    mlflow.log_metric("accuracy", accuracyNN)
    mlflow.log_metric("precision", precisionNN)
    mlflow.log_metric("recall", recallNN)
    mlflow.log_metric("f1", f1NN)  
    
    # Almacenar función de costo del modelo
    
    mlflow.log_figure(plot_training_history(history), "lossNN.png")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("model_type", "Red Neuronal Keras")

    # Infer the model signature
    signature = infer_signature(X_trainNN, modeloKeras.predict(X_trainNN))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=modeloKeras,
        artifact_path="matematicas-modelNN",
        signature=signature,
        input_example=X_trainNN,
        registered_model_name="matematicas-Keras",
    )

# Start an MLflow run DecisionTree
with mlflow.start_run():

    # Log the hyperparameters
    mlflow.log_params(params)

    # Métricas de precision del modelo
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)  

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("model_type","DecisionTree")

    # Infer the model signature
    signature = infer_signature(X_train, modelo.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=modelo,
        artifact_path="matematicas-model",
        signature=signature,
        input_example=X_train,
        registered_model_name="matematicas-decision-tree",
    )