import numpy as np
import torch


def compute_ltrd(
    y_true,
    y_pred,
    num_classes,
    lambda_bias=0.8,
    return_details=False,
):
    """
    
    Long-Tail Rank Deviation
    Parameters
    ----------
    y_true : array-like or torch.Tensor
        Ground truth labels (N,)
    y_pred : array-like or torch.Tensor
        Predicted labels (N,)
    num_classes : int
        Total number of classes
    lambda_bias : float
        Directional bias penalty coefficient (λ ≥ 0)
    return_details : bool
        If True, return per-class RE_c and weights

    -------
   
    """

    # ==============================
    # Convert to numpy
    # ==============================
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    N = len(y_true)
    assert N > 0

    # ==============================
    # Class statistics
    # ==============================
    class_counts = np.bincount(y_true, minlength=num_classes)
    class_freq = class_counts / N

    # Avoid divide-by-zero
    eps = 1e-12
    weights = 1.0 / np.sqrt(class_freq + eps)

    RE_c_list = []
    valid_weights = []

    # ==============================
    # Compute class-wise RE_c
    # ==============================
    for c in range(num_classes):

        idx = np.where(y_true == c)[0]
        Nc = len(idx)

        if Nc == 0:
            continue

        y_c = y_true[idx]
        yhat_c = y_pred[idx]

        # Ordinal distance term
        ordinal_error = np.mean(np.abs(y_c - yhat_c))

        # Directional bias term
        bias = np.mean(yhat_c - y_c)
        bias_penalty = np.abs(bias)

        RE_c = ordinal_error + lambda_bias * bias_penalty

        RE_c_list.append(RE_c)
        valid_weights.append(weights[c])

    RE_c_list = np.array(RE_c_list)
    valid_weights = np.array(valid_weights)

    # ==============================
    # Final LTRD
    # ==============================
    ltre_db = np.sum(valid_weights * RE_c_list) / np.sum(valid_weights)

    if return_details:
        return ltre_db, {
            "RE_c": RE_c_list,
            "weights": valid_weights,
            "class_counts": class_counts,
        }

    return ltre_db
