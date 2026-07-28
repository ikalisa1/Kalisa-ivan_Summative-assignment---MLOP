"""
Prediction Script
Handles data point insertion and model predictions
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    def __init__(self, model_path: str | None = None, scaler_path: str | None = None):
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else (self.project_root / "models" / "iris_model.pkl")
        self.scaler_path = Path(scaler_path) if scaler_path else (self.project_root / "models" / "scaler.pkl")
        if not self.model_path.is_absolute():
            self.model_path = self.project_root / self.model_path
        if not self.scaler_path.is_absolute():
            self.scaler_path = self.project_root / self.scaler_path
        self.model = None
        self.scaler = None
        self.class_names = ["setosa", "versicolor", "virginica"]
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        try:
            if not self.model_path.exists():
                keras_path = self.project_root / "models" / "iris_model.keras"
                legacy_h5_path = self.project_root / "models" / "iris_model.h5"
                legacy_pickle_path = self.project_root / "models" / "iris_model.pkl"
                if legacy_pickle_path.exists():
                    self.model_path = legacy_pickle_path
                elif keras_path.exists():
                    self.model_path = keras_path
                elif legacy_h5_path.exists():
                    self.model_path = legacy_h5_path
                else:
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")

            if self.model_path.suffix in {".keras", ".h5"}:
                from tensorflow import keras

                self.model = keras.models.load_model(self.model_path, compile=False)
            else:
                scripts_path = str(self.project_root / "scripts")
                if scripts_path not in sys.path:
                    sys.path.insert(0, scripts_path)
                # Required for loading legacy pickled classifier instances.
                import model_utils  # noqa: F401
                with self.model_path.open("rb") as file_handle:
                    self.model = pickle.load(file_handle)
            logger.info("Model loaded successfully")

            with self.scaler_path.open("rb") as file_handle:
                self.scaler = pickle.load(file_handle)
            logger.info("Scaler loaded successfully")
        except Exception as exc:
            logger.error(f"Error loading model/scaler: {exc}")
            raise

    def _predict_probabilities(self, features_scaled: list[list[float]]) -> list[list[float]]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features_scaled)

        prediction_probs = self.model.predict(np.asarray(features_scaled, dtype="float32"), verbose=0)
        return prediction_probs.tolist()

    def predict_from_csv_row(self, features):
        try:
            features = [[float(value) for value in features]]
            features_scaled = self.scaler.transform(features)
            prediction_probs = self._predict_probabilities(features_scaled)
            predicted_class = max(range(len(prediction_probs[0])), key=prediction_probs[0].__getitem__)
            predicted_name = self.class_names[predicted_class]
            confidence = float(prediction_probs[0][predicted_class]) * 100

            logger.info("Prediction made")
            logger.info(f"  Input features: {features[0]}")
            logger.info(f"  Predicted class: {predicted_name}")
            logger.info(f"  Confidence: {confidence:.2f}%")

            return predicted_name, confidence, prediction_probs[0]
        except Exception as exc:
            logger.error(f"Error in prediction: {exc}")
            raise

    def predict_from_csv_file(self, filepath):
        try:
            csv_path = Path(filepath)
            if not csv_path.is_absolute():
                csv_path = self.project_root / csv_path

            with csv_path.open(newline="", encoding="utf-8") as file_handle:
                reader = csv.DictReader(file_handle)
                rows = list(reader)

            logger.info(f"CSV file loaded: {len(rows)} samples")
            feature_rows = [
                [
                    float(row["sepal_length"]),
                    float(row["sepal_width"]),
                    float(row["petal_length"]),
                    float(row["petal_width"]),
                ]
                for row in rows
            ]
            scaled_rows = self.scaler.transform(feature_rows)
            prediction_probs = self._predict_probabilities(scaled_rows)
            predictions = [max(range(len(row)), key=row.__getitem__) for row in prediction_probs]
            confidences = [max(row) * 100 for row in prediction_probs]

            results = []
            for row, prediction, confidence in zip(rows, predictions, confidences):
                result = dict(row)
                result["predicted_class"] = self.class_names[prediction]
                result["confidence"] = confidence
                results.append(result)

            logger.info(f"Predictions completed for {len(results)} samples")
            return results
        except Exception as exc:
            logger.error(f"Error in batch prediction: {exc}")
            raise

    def predict_from_dict(self, data_dict):
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        features = [data_dict.get(name) for name in feature_names]
        if None in features:
            raise ValueError(f"Missing required features. Need: {feature_names}")
        return self.predict_from_csv_row(features)

    def get_prediction_report(self, features):
        predicted_name, confidence, probs = self.predict_from_csv_row(features)
        return {
            "predicted_class": predicted_name,
            "confidence": confidence,
            "probability_distribution": {
                self.class_names[i]: float(probs[i]) * 100 for i in range(len(self.class_names))
            },
            "features": {
                "sepal_length": features[0],
                "sepal_width": features[1],
                "petal_length": features[2],
                "petal_width": features[3],
            },
        }


def main():
    engine = PredictionEngine()

    logger.info("\n--- PREDICTION EXAMPLES ---\n")
    features = [5.1, 3.5, 1.4, 0.2]
    predicted_class, confidence, _ = engine.predict_from_csv_row(features)
    print(f"Input: {features}")
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f}%)\n")

    logger.info("Example 2: Detailed Prediction Report")
    report = engine.get_prediction_report([6.3, 2.9, 5.6, 1.8])
    print(json.dumps(report, indent=2))

    logger.info("\nExample 3: Batch Prediction from CSV")
    results = engine.predict_from_csv_file("data/iris_test.csv")
    for row in results[:3]:
        print(row)

    logger.info("\nPREDICTION ENGINE READY FOR DEPLOYMENT")


if __name__ == "__main__":
    main()