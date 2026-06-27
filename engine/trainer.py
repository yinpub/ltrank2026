import torch

from torch.cuda.amp import autocast

def train_one_epoch(model, loader, optimizer, criterion, device,
                    sup=False, scaler=None, use_amp=False,orl=False):

    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # =========================
        # AMP Forward
        # =========================
        with autocast(enabled=use_amp):

            embedding, logits = model(images)

            if not sup:
                loss = criterion(logits, labels)
            
            else:
                loss = criterion(embedding, logits, labels)

        # =========================
        # AMP Backward
        # =========================
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss
