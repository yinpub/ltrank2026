import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from utils.metrics import compute_ltrd

import torch.nn.functional as F

def evaluate(
    model,
    loader,
    device,
    criterion=None,          
    class_names=None,
    compute_centroid=None,
    compute_bvr=True
):
    model.eval()

    all_preds = []
    all_targets = []
    all_embeddings = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            embedding, logits = model(images)

            # normalize
            embedding = F.normalize(embedding, dim=1)

            all_embeddings.append(embedding.cpu())

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # =========================
    # metrics
    # =========================
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(all_embeddings) == 0:
        print("No embeddings collected.")
        return None, None

    all_embeddings = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.tensor(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    ltre = compute_ltrd(
        y_true=all_targets,
        y_pred=all_preds,
        num_classes=len(class_names),
        lambda_bias=0.8
    )

    mae = np.mean(np.abs(all_targets - all_preds))

    print("\n===== Evaluation =====")
    print(f"LTRE-DB: {ltre:.4f}")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"MAE: {mae:.4f}")
    print("Confusion Matrix:\n", cm)

    if class_names is not None:
        print("\nPer-class metrics:")
        print(classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            digits=4
        ))

    # # =====================================================
    # # 1️⃣ Centroid Distance
    # # =====================================================
    # if compute_centroid and class_names is not None:

    #     print("\n===== Centroid Distance =====")

    #     num_classes = len(class_names)
    #     centroids = []

    #     for k in range(num_classes):
    #         mask = (labels_tensor == k)
    #         class_embeddings = all_embeddings[mask]

    #         if len(class_embeddings) == 0:
    #             centroids.append(None)
    #             continue

    #         centroid = class_embeddings.mean(dim=0)
    #         centroid = F.normalize(centroid.unsqueeze(0), dim=1).squeeze(0)
    #         centroids.append(centroid)

    #     for k in range(num_classes - 1):
    #         if centroids[k] is None or centroids[k+1] is None:
    #             continue

    #         dist = torch.norm(centroids[k] - centroids[k+1], p=2)
    #         print(f"D({k}-{k+1}) = {dist.item():.4f}")

    #     # Relative separation ratio
    #     if num_classes >= 3:
    #         if centroids[0] is not None and centroids[1] is not None and centroids[2] is not None:
    #             d01 = torch.norm(centroids[0] - centroids[1], p=2)
    #             d12 = torch.norm(centroids[1] - centroids[2], p=2)
    #             ratio = d12 / (d01 + 1e-8)
    #             print(f"Relative Separation Ratio (1-2 / 0-1): {ratio:.4f}")

    # # =====================================================
    # # 2️⃣ Boundary Violation Rate（δ ）
    # # =====================================================
    # if compute_bvr and criterion is not None and class_names is not None:

    #     print("\n===== Boundary Violation Rate (BVR) =====")

    #     if hasattr(criterion, "fmmoc"):
    #         effective_margins = criterion.fmmoc.get_effective_margins()
    #     else:
    #         print("Criterion does not contain FMMOC.")
    #         effective_margins = None

    #     if effective_margins is not None:

    #         num_classes = len(class_names)

    #         for h in range(num_classes - 1):

    #             mask_h = (labels_tensor == h)
    #             mask_hp1 = (labels_tensor == h + 1)

    #             emb_h = all_embeddings[mask_h]
    #             emb_hp1 = all_embeddings[mask_hp1]

    #             if len(emb_h) == 0 or len(emb_hp1) == 0:
    #                 continue

               
    #             dist_matrix = torch.cdist(emb_h, emb_hp1, p=2)

    #             min_dist, _ = dist_matrix.min(dim=1)

    #             tau = effective_margins[h]

    #             print("Example min_dist:", min_dist.mean().item())
    #             print("Tau:", tau)

    #             violations = (min_dist < tau).float()
    #             bvr = violations.mean().item()

    #             print(f"BVR({h}-{h+1}) = {bvr:.4f}")

    return ltre, acc








def evaluatereg(model, loader, device, class_names=None):
    model.eval()

    all_preds = []
    all_targets = []

    num_classes = len(class_names)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            embedding, outputs = model(images)  # outputs = (B,1)

            preds = outputs.squeeze(1)          # (B,)
            preds = torch.round(preds)          # regression → discrete
            preds = preds.clamp(0, num_classes - 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    ltre = compute_ltrd(
        y_true=all_targets,
        y_pred=all_preds,
        num_classes=num_classes,
        lambda_bias=0.8
    )
    mae = np.mean(np.abs(all_targets - all_preds))

    print("\n===== Evaluation =====")
    print(f"LTRE-DB: {ltre:.4f}")
    print("Overall Accuracy:", acc)
    print(f"MAE: {mae:.4f}")
    print("Confusion Matrix:\n", cm)

    if class_names is not None:
        print("\nPer-class metrics:")
        print(
            classification_report(
                all_targets,
                all_preds,
                target_names=class_names,
                digits=4
            )
        )

    return ltre, acc
