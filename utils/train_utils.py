import torch
from tqdm import tqdm

def train(model, loader, criterion, optimizer, device, writer=None, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0

    loop = tqdm(
        enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{total_epochs}"
    )

    for i, (x, y) in loop:
        x = x.to(device)           # 입력 이미지 [B,C,H,W]
        y = y.to(device).long()  # 정답 마스크 [B, H, W]

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TensorBoard
        if writer is not None and epoch is not None:
            global_step = epoch * len(loader) + i
            writer.add_scalar("Loss/batch", loss.item(), global_step)

        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_total = 0.0
    num_batches = 0

    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), desc="[Evaluating]", leave=False)

        for batch_idx, (x, y) in loop:
            x = x.to(device)
            y = y.to(device).long()  

            preds = model(x)  # [B, C, H, W]
            loss = criterion(preds, y)
            running_loss += loss.item()

            # softmax -> 예측 클래스 만들기
            probs = torch.softmax(preds, dim=1)
            preds_cls = torch.argmax(probs, dim=1) # 가장 큰 값의 index 선택택

            # One-hot encoding
            num_classes = preds.shape[1] # 클래스 수
            
            preds_onehot = torch.nn.functional.one_hot(preds_cls, num_classes=num_classes)
            y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes)
            
            preds_onehot = preds_onehot.permute(0, 3, 1, 2).float()
            y_onehot = y_onehot.permute(0, 3, 1, 2).float()

            # IoU 계산 
            intersection = (preds_onehot * y_onehot).sum(dim=(2, 3))  # [B, C]
            union = (
                preds_onehot.sum(dim=(2, 3)) + y_onehot.sum(dim=(2, 3)) - intersection
            )

            # 클래스별 IoU 계산 (background 클래스 제외)
            iou = (intersection + 1e-7) / (union + 1e-7)  # [B, C]

            # Background 클래스(0번) 제외하고 평균 계산
            if num_classes > 1:
                iou_foreground = iou[:, 1:]  # background 제외
                mean_iou_per_batch = iou_foreground.mean(dim=1)  # [B]
            else:
                mean_iou_per_batch = iou.mean(dim=1)  # [B]

            mean_iou = mean_iou_per_batch.mean().item()

            iou_total += mean_iou
            num_batches += 1

            loop.set_postfix(val_loss=loss.item(), val_iou=mean_iou)

    avg_loss = running_loss / num_batches
    avg_iou = iou_total / num_batches
    return avg_loss, avg_iou
