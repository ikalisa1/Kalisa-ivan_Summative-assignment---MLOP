"""Lightweight model utilities for Iris classification."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


class IrisStandardizer:
    def __init__(self) -> None:
        self.means: list[float] = []
        self.scales: list[float] = []

    def fit(self, rows: list[list[float]]) -> "IrisStandardizer":
        columns = list(zip(*rows))
        self.means = [sum(column) / len(column) for column in columns]
        self.scales = []
        for index, column in enumerate(columns):
            mean = self.means[index]
            variance = sum((value - mean) ** 2 for value in column) / max(len(column) - 1, 1)
            self.scales.append(math.sqrt(variance) or 1.0)
        return self

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [
            [(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)]
            for row in rows
        ]

    def fit_transform(self, rows: list[list[float]]) -> list[list[float]]:
        return self.fit(rows).transform(rows)


@dataclass
class IrisCentroidClassifier:
    centroids: dict[int, list[float]] = field(default_factory=dict)
    class_labels: list[int] = field(default_factory=lambda: [0, 1, 2])

    def fit(self, rows: list[list[float]], labels: list[int]) -> "IrisCentroidClassifier":
        grouped: dict[int, list[list[float]]] = {label: [] for label in self.class_labels}
        for row, label in zip(rows, labels):
            grouped[int(label)].append(row)

        self.centroids = {}
        for label, samples in grouped.items():
            if not samples:
                continue
            self.centroids[label] = [sum(values) / len(values) for values in zip(*samples)]
        return self

    def _distance(self, left: list[float], right: list[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        probabilities: list[list[float]] = []
        ordered_labels = self.class_labels
        for row in rows:
            distances = [self._distance(row, self.centroids[label]) for label in ordered_labels]
            scores = [math.exp(-distance) for distance in distances]
            total = sum(scores) or 1.0
            probabilities.append([score / total for score in scores])
        return probabilities

    def predict(self, rows: list[list[float]]) -> list[int]:
        probability_rows = self.predict_proba(rows)
        return [self.class_labels[max(range(len(row)), key=row.__getitem__)] for row in probability_rows]