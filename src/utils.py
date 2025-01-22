from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss
)
import numpy as np

def evaluate_predictions(y_true, y_pred, y_prob=None, average=None, task_type='auto'):
    """
    Comprehensive evaluation function for classification tasks.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.
    - y_prob: array-like of shape (n_samples,) or (n_samples, n_classes), default=None
        Predicted probabilities for the positive class (binary) or all classes (multi-class).
    - average: str, default=None
        Averaging method for metrics ('binary', 'micro', 'macro', 'weighted').
        Automatically set to 'binary' for binary classification if not specified.
    - task_type: str, default='auto'
        Specifies the task type ('binary', 'multiclass', or 'auto').
        If 'auto', the task type is inferred from the labels.

    Returns:
    - metrics_dict: dict
        A dictionary containing various evaluation metrics.
    """
    # Infer task type if set to 'auto'
    unique_classes = np.unique(y_true)
    if task_type == 'auto':
        task_type = 'binary' if len(unique_classes) <= 2 else 'multiclass'

    # Set default averaging based on task type
    if average is None:
        average = 'binary' if task_type == 'binary' else 'macro'

    metrics_dict = {}

    # Basic metrics
    metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metrics_dict['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics_dict['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics_dict['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics_dict['confusion_matrix'] = cm

    # ROC AUC and Log Loss
    if y_prob is not None:
        if task_type == 'binary':
            metrics_dict['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:  # Multiclass case
            metrics_dict['roc_auc'] = roc_auc_score(y_true, y_prob, average=average, multi_class='ovr')
        metrics_dict['log_loss'] = log_loss(y_true, y_prob)

    # Classification report
    metrics_dict['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return metrics_dict

# Example usage
if __name__ == "__main__":
    # Sample binary data
    y_true_binary = [0, 1, 1, 0, 1, 0, 1]
    y_pred_binary = [0, 1, 1, 0, 0, 0, 1]
    y_prob_binary = [0.1, 0.8, 0.9, 0.2, 0.4, 0.3, 0.7]

    metrics_binary = evaluate_predictions(y_true_binary, y_pred_binary, y_prob=y_prob_binary, task_type='binary')
    print("Binary Classification Metrics:")
    for key, value in metrics_binary.items():
        print(f"{key}: {value}")

    # Sample multi-class data
    y_true_multiclass = [0, 1, 2, 0, 1, 2, 1]
    y_pred_multiclass = [0, 1, 2, 0, 0, 2, 1]
    y_prob_multiclass = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.4, 0.5, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2]
    ]

    metrics_multiclass = evaluate_predictions(y_true_multiclass, y_pred_multiclass, y_prob=y_prob_multiclass, task_type='multiclass')
    print("\nMulti-class Classification Metrics:")
    for key, value in metrics_multiclass.items():
        print(f"{key}: {value}")
