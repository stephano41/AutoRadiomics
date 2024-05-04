from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import logging

log = logging.getLogger(__name__)


def roc_auc(y_true, y_pred, average='macro', multi_class='raise', labels=None):
    if len(y_pred.shape) >= 2: #if given in n_sample x n_class format
        if y_pred.shape[1] == 2: #if it's binary case
            y_pred = y_pred[:, 1]
        elif y_pred.shape[1] == 1: #if it's binary case
            y_pred = y_pred[:, 0]
    if isinstance(y_true, tuple) and len(y_true)==5:
        # account for encoder models that pass multiple forward
        y_true=y_true[4]

    try:
        auc_score = roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class, labels=labels)
    except ValueError as e:
        if 'Only one class present' not in str(e):
            raise e
        log.error("Only one class present in y_true. ROC AUC score is not defined in that case")
        auc_score = np.nan
    return auc_score


def pr_auc(y_true, y_pred, average='macro', multi_class='raise', labels=None):
    if len(y_pred.shape) >= 2: #if given in n_sample x n_class format
        if y_pred.shape[1] == 2: #if it's binary case
            y_pred = y_pred[:, 1]
        elif y_pred.shape[1] == 1: #if it's binary case
            y_pred = y_pred[:, 0]

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auc_score = auc(recall, precision)
    except ValueError as e:
        if 'Only one class present' not in str(e):
            raise e
        log.error("Only one class present in y_true. PR AUC score is not defined in that case")
        auc_score = np.nan
    return auc_score
