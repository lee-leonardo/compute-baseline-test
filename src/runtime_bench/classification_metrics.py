"""Classification quality independent of runtime, computed outside timed batches."""

import numpy as np


def summarize(confusion, labels):
    """Summarize integer counts with rows=true class and columns=predicted class.

    Undefined precision/recall are reported as null, not fabricated measurements.
    Macro F1 uses zero for classes with no true positives, over all known labels.
    Balanced accuracy averages recall only for classes present in evaluation.
    """
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)) or (matrix < 0).any() or not matrix.sum():
        raise ValueError("Expected a nonempty square confusion matrix matching labels")
    support, predicted = matrix.sum(1), matrix.sum(0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros(len(labels)), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(labels)), where=support > 0)
    denominator = support + predicted
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(len(labels)), where=denominator > 0)
    return {
        "metrics_version": "classification-quality-v1",
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted; label order matches per_class",
        "evaluation_rows": int(support.sum()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "missing_evaluation_labels": [label for label, n in zip(labels, support) if not n],
        "per_class": [
            {
                "label": label,
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": float(precision[i]) if predicted[i] else None,
                "recall": float(recall[i]) if support[i] else None,
                "f1": float(f1[i]),
            }
            for i, label in enumerate(labels)
        ],
    }
