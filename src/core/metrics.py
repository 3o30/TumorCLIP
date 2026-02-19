import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)


class MetricsCalculator:

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision_recall_f1(y_true, y_pred, average='weighted'):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        return precision, recall, f1

    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def calculate_per_class_metrics(y_true, y_pred, class_names):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics = {}
        for i, class_name in enumerate(class_names):
            metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }

        return metrics

    @staticmethod
    def calculate_all_metrics(y_true, y_pred, class_names=None):
        metrics = {}

        # Global accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # Weighted precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)

        # Confusion matrix (converted to list for JSON compatibility)
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Per-class metrics (if class names provided)
        if class_names:
            per_class_metrics = MetricsCalculator.calculate_per_class_metrics(
                y_true, y_pred, class_names
            )
            metrics['per_class'] = per_class_metrics

        # Full classification report (if class names provided)
        if class_names:
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            # Convert NumPy types to native Python types
            metrics['classification_report'] = MetricsCalculator._convert_report_types(report)

        return metrics

    @staticmethod
    def _convert_report_types(report):
        converted = {}
        for key, value in report.items():
            if isinstance(value, dict):
                converted[key] = MetricsCalculator._convert_report_types(value)
            elif isinstance(value, np.number):
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted
