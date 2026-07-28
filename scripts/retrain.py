"""
Retraining Script
Handles data uploading, database saving, preprocessing, and model retraining
"""

from __future__ import annotations

import csv
import pickle
import logging
from pathlib import Path
from datetime import datetime
import sqlite3
import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    def __init__(self, db_path: str | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path or (self.data_dir / "training_history.db"))
        self.init_database()

    def init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                data_batch BLOB,
                feature_count INTEGER,
                sample_count INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                accuracy REAL,
                loss REAL,
                f1_score REAL,
                precision REAL,
                recall REAL,
                epochs_trained INTEGER
            )
            """
        )
        conn.commit()
        conn.close()
        logger.info("✓ Database initialized")

    def upload_and_save_data(self, filepath: str):
        try:
            csv_path = Path(filepath)
            if not csv_path.is_absolute():
                csv_path = self.project_root / csv_path

            with csv_path.open(newline="", encoding="utf-8") as file_handle:
                reader = csv.DictReader(file_handle)
                rows = list(reader)

            logger.info(f"✓ Data uploaded: {len(rows)} samples")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO training_data
                (timestamp, data_batch, feature_count, sample_count)
                VALUES (?, ?, ?, ?)
                """,
                (datetime.now(), pickle.dumps(rows), len(reader.fieldnames or []), len(rows)),
            )
            conn.commit()
            last_row_id = cursor.lastrowid
            conn.close()
            logger.info(f"✓ Data saved to database (ID: {last_row_id})")

            return rows
        except Exception as exc:
            logger.error(f"Error uploading data: {exc}")
            raise

    def preprocess_data(self, rows):
        try:
            feature_rows: list[list[float]] = []
            labels: list[int] = []
            label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}

            for row in rows:
                feature_rows.append(
                    [
                        float(row["sepal_length"]),
                        float(row["sepal_width"]),
                        float(row["petal_length"]),
                        float(row["petal_width"]),
                    ]
                )
                labels.append(label_map[str(row["species"]).strip().lower()])

            logger.info(f"✓ Data prepared: {len(feature_rows)} samples")
            return feature_rows, labels
        except Exception as exc:
            logger.error(f"Error in preprocessing: {exc}")
            raise

    def build_model(self):
        model = keras.Sequential(
            [
                layers.Input(shape=(4,)),
                layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.3),
                layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def retrain_model(self, X_train, y_train, model_path: str | None = None, epochs: int = 1):
        try:
            model_path = Path(model_path) if model_path else (self.models_dir / "iris_model.keras")
            if not model_path.is_absolute():
                model_path = self.project_root / model_path

            logger.info(f"Loading pre-trained model from: {model_path}")
            try:
                model = keras.models.load_model(model_path)
                logger.info("✓ Pre-trained model loaded successfully")
            except FileNotFoundError:
                logger.warning("Pre-trained model not found, creating new model")
                model = self.build_model()
            except Exception as exc:
                logger.warning(f"Could not load pre-trained model ({exc}); creating new model")
                model = self.build_model()

            logger.info("Starting retraining...")
            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            history = model.fit(
                np.asarray(X_train, dtype="float32"),
                np.asarray(y_train),
                validation_split=0.2,
                epochs=epochs,
                batch_size=8,
                callbacks=[early_stop],
                verbose=0,
            )
            logger.info(f"✓ Model retrained for {len(history.history['loss'])} epochs")

            model.save(model_path)
            logger.info(f"✓ Retrained model saved to: {model_path}")

            return model, history
        except Exception as exc:
            logger.error(f"Error in retraining: {exc}")
            raise

    def save_metrics(self, accuracy, loss, f1_score, precision, recall, epochs):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO training_metrics
            (timestamp, accuracy, loss, f1_score, precision, recall, epochs_trained)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(), accuracy, loss, f1_score, precision, recall, epochs),
        )
        conn.commit()
        conn.close()
        logger.info("✓ Metrics saved to database")


def main():
    pipeline = RetrainingPipeline()

    logger.info("\n--- STEP 1: DATA UPLOADING & DATABASE SAVING ---")
    rows = pipeline.upload_and_save_data("data/iris_train.csv")

    logger.info("\n--- STEP 2: DATA PREPROCESSING ---")
    X, y = pipeline.preprocess_data(rows)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("✓ Features standardized with StandardScaler")

    with (pipeline.models_dir / "scaler.pkl").open("wb") as file_handle:
        pickle.dump(scaler, file_handle)
    logger.info("✓ Scaler saved for prediction use")

    logger.info("\n--- STEP 3: RETRAINING WITH PRE-TRAINED MODEL ---")
    model, history = pipeline.retrain_model(X_train_scaled, y_train, epochs=100)

    probabilities = model.predict(np.asarray(X_test_scaled, dtype="float32"), verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    loss = float(model.evaluate(np.asarray(X_test_scaled, dtype="float32"), np.asarray(y_test), verbose=0)[0])
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1_metric = f1_score(y_test, predictions, average="weighted")

    pipeline.save_metrics(
        accuracy=accuracy,
        loss=loss,
        f1_score=f1_metric,
        precision=precision,
        recall=recall,
        epochs=len(history.history["loss"]),
    )

    logger.info("\n✓ RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"  Final Accuracy: {accuracy:.4f}")
    logger.info(f"  Final Loss: {loss:.4f}")
    logger.info(f"  Final F1 Score: {f1_metric:.4f}")


if __name__ == "__main__":
    main()