import numpy as np
from sklearn.metrics import roc_curve, auc


def accuracy_with_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    # if sigmoid:
    #     y_pred = y_pred.sigmoid()
    # return np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1).sum()
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).byte()  # boolean tensor to uint8 tensor
    return np.mean((y_pred == y_true.byte()).float().numpy(), axis=1).mean()


def roc_auc(all_labels, all_logits, num_labels):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    all_labels = all_labels.astype(int)
    for i in range(num_labels):
        if all_labels[:, i].sum() == 0 or all_logits[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    if all_labels.sum() != 0 and all_logits.sum() != 0:
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc
