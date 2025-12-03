# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def load_and_clean_data(filepath):
    """Paso 1: Carga y limpieza de datos"""
    df = pd.read_csv(filepath, compression="zip")

    # Renombrar columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    df = df.drop(columns=["ID"])

    # Eliminar registros con información no disponible (EDUCATION=0 o MARRIAGE=0)
    df = df[df["EDUCATION"] != 0]
    df = df[df["MARRIAGE"] != 0]

    # Agrupar EDUCATION > 4 en categoría "others" (valor 4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


def split_data(df):
    """Paso 2: Dividir en X e y"""
    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


def create_pipeline():
    """Paso 3: Crear pipeline"""
    # Columnas categóricas para one-hot encoding
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    # Preprocesador: OneHotEncoder para variables categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop=None, sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",
    )

    # Pipeline completo
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("scaler", StandardScaler()),
            ("selectkbest", SelectKBest(score_func=f_classif)),
            ("svc", SVC()),
        ]
    )

    return pipeline


def optimize_model(pipeline, x_train, y_train):
    """Paso 4: Optimizar hiperparámetros con GridSearchCV"""
    param_grid = {
        "pca__n_components": [15, 20],
        "selectkbest__k": [15, 20],
        "svc__kernel": ["rbf"],
        "svc__C": [10, 50, 100],
        "svc__gamma": ["scale", 0.01],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(x_train, y_train)

    return grid_search


def save_model(model, filepath):
    """Paso 5: Guardar modelo comprimido"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, "wb") as f:
        pickle.dump(model, f)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """Paso 6 y 7: Calcular métricas y matrices de confusión"""
    metrics = []

    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Métricas para train
    metrics.append({
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    })

    # Métricas para test
    metrics.append({
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    })

    # Matriz de confusión para train
    cm_train = confusion_matrix(y_train, y_train_pred)
    metrics.append({
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
    })

    # Matriz de confusión para test
    cm_test = confusion_matrix(y_test, y_test_pred)
    metrics.append({
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
    })

    return metrics


def save_metrics(metrics, filepath):
    """Guardar métricas en archivo JSON (una línea por diccionario)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")


def main():
    # Paso 1: Cargar y limpiar datos
    train_df = load_and_clean_data("files/input/train_data.csv.zip")
    test_df = load_and_clean_data("files/input/test_data.csv.zip")

    # Paso 2: Dividir en X e y
    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    # Paso 3: Crear pipeline
    pipeline = create_pipeline()

    # Paso 4: Optimizar modelo
    model = optimize_model(pipeline, x_train, y_train)

    # Paso 5: Guardar modelo
    save_model(model, "files/models/model.pkl.gz")

    # Paso 6 y 7: Calcular y guardar métricas
    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics, "files/output/metrics.json")

    return model


if __name__ == "__main__":
    main()
#
