import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Frequency-aware Multi-Margin Loss Ordinal classification
# =====================================================
class FMMOC_Loss(nn.Module):
    def __init__(
        self,
        n_classes,
        class_counts,
        device,
        gamma=0.5,      # anchor-level exponent
        delta=0.5,      # boundary-level exponent
        single_margin=False,
        normalize_beta=True,
    ):
        super().__init__()

        self.device = device
        self.n_classes = n_classes
        self.n_margins = n_classes - 1
        self.single_margin = single_margin

        self.gamma = gamma
        self.delta = delta
        self.normalize_beta = normalize_beta

        # ---------- class frequency ----------
        class_counts = torch.tensor(class_counts, dtype=torch.float32).to(device)
        self.register_buffer("class_counts", class_counts)

        # ---------- boundary frequency ----------
        # boundary h between class h and h+1
        boundary_counts = class_counts[:-1] + class_counts[1:]
        self.register_buffer("boundary_counts", boundary_counts)

        # ---------- margins ----------
        if single_margin:
            self.margin = nn.Parameter(torch.tensor(0.5))
        else:
            self.margins = nn.Parameter(torch.ones(self.n_margins) * 0.5)


    def forward(self, embedding, target):

        embedding = F.normalize(embedding, dim=1)
        sim_matrix = torch.matmul(embedding, embedding.T)

        B = embedding.size(0)

        # ----------------------------
        # positive / negative masks
        # ----------------------------
        label_diff = torch.abs(target.unsqueeze(0) - target.unsqueeze(1))

        positives = label_diff == 0
        negatives = ~positives
        positives.fill_diagonal_(False)

        # ----------------------------
        # anchor-level reweight α(y)
        # ----------------------------

        if self.gamma > 0:
            alpha = (1.0 / self.class_counts[target]) ** self.gamma
        else:
            alpha = torch.ones_like(target, dtype=torch.float32)

        # ----------------------------
        # boundary-level β_h scaling
        # ----------------------------
        if self.delta > 0:
            beta = (1.0 / self.boundary_counts) ** self.delta

            if self.normalize_beta:
                beta = beta / beta.mean()#.to(self.device)

        else:
            beta = torch.ones_like(self.boundary_counts)#.to(self.device)

        # ----------------------------
        # compute margins
        # ----------------------------
        if self.single_margin:
            base_margins = F.softplus(self.margin).repeat(self.n_margins).to(self.device)
        else:
            base_margins = F.softplus(self.margins).to(self.device)

        # apply boundary scaling
        scaled_margins = beta * base_margins

        class_pos = torch.cumsum(
            torch.cat([torch.zeros(1, device=self.device), scaled_margins]),
            dim=0,
        )

        distance_matrix = torch.abs(
            class_pos.unsqueeze(0) - class_pos.unsqueeze(1)
        )

        # ----------------------------
        # assign margins per pair
        # ----------------------------
        label_i = target.unsqueeze(0).repeat(B, 1)
        pair_margins = distance_matrix[label_i, label_i.T]
        pair_margins[~negatives] = 0.0

        # ----------------------------
        # N-pair loss
        # ----------------------------
        loss = 0.0
        count = 0

        for i in range(B):

            pos_idx = positives[i]
            neg_idx = negatives[i]

            if pos_idx.sum() == 0 or neg_idx.sum() == 0:
                continue

            pos_sim = sim_matrix[i][pos_idx]
            neg_sim = sim_matrix[i][neg_idx]
            neg_margin = pair_margins[i][neg_idx]

            diff = (
                neg_sim.unsqueeze(0)
                + neg_margin.unsqueeze(0)
                - pos_sim.unsqueeze(1)
            )

            loss_i = F.relu(diff).mean()

            # apply anchor scaling α(y)
            loss_i = alpha[i] * loss_i

            loss += loss_i
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)

        return loss / count


# =====================================================
# FMML + CE
# =====================================================
class FMMOC_TotalLoss(nn.Module):
    def __init__(
        self,
        n_classes,
        class_counts,
        device,
        gamma=0.5,
        delta=0.5,
        single_margin=False,
        scale=0.5,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.fmmoc = FMMOC_Loss(
            n_classes=n_classes,
            class_counts=class_counts,
            device=device,
            gamma=gamma,
            delta=delta,
            single_margin=single_margin,
        )
        self.scale = scale
        #self.mse = nn.MSELoss()
        #self.scale2 = 0.05
        self.register_buffer("class_indices",torch.arange(n_classes, dtype=torch.float32).to(device))

    def forward(self, embedding, logits, target):

        loss_ce = self.ce(logits, target)
        loss_mm = self.fmmoc(embedding, target)

        probs = F.softmax(logits, dim=1)

        # expected rank prediction
        y_pred = torch.sum(
            probs * self.class_indices.unsqueeze(0),
            dim=1
        )

        # loss_mse = self.mse(
        #     y_pred,
        #     target.float()
        # )

        return loss_ce + self.scale * loss_mm   #+ self.scale2 * loss_mse
