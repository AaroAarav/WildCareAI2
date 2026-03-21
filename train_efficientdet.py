import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from utils import set_seeds, save_hyperparams, setup_environment, DEVICE

# ── Transforms (Albumentations) ───────────────────────────────────────────────
def get_transform(is_train):
    # Added clip=True to both BboxParams to prevent out-of-bounds floating point crashes
    if is_train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], clip=True))
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], clip=True))

# ── Dataset ───────────────────────────────────────────────────────────────────
class EffDetDataset(Dataset):
    def __init__(self, split):
        self.root = f"dataset/images/{split}"
        self.coco = COCO(f"coco_annotations/instances_{split}.json")
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = get_transform(is_train=(split == 'train'))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        # Load Image using OpenCV
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_h, img_w = image.shape[:2]

        boxes = []
        labels = []
        
        for ann in coco_annotation:
            x, y, w, h = ann['bbox']
            
            # 1. Clamp slightly negative floating point coordinates back to 0.0
            x = max(0.0, float(x))
            y = max(0.0, float(y))
            
            # 2. Ensure the width and height don't push the box outside the right/bottom edges
            w = min(float(w), img_w - x)
            h = min(float(h), img_h - y)

            # Safety check: if the box is too small or negative, skip it
            if w <= 1.0 or h <= 1.0:
                continue
                
            boxes.append([x, y, w, h])
            # CRITICAL: EfficientDet reserves 0 for background. Shift classes +1.
            labels.append(ann['category_id'] + 1) 

        # Apply resizing and normalization
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        # EfficientDet requires boxes in [y_min, x_min, y_max, x_max] format
        effdet_boxes = []
        for b in boxes:
            x_min, y_min, w, h = b
            effdet_boxes.append([y_min, x_min, y_min + h, x_min + w])

        target = {
            'bbox': torch.tensor(effdet_boxes, dtype=torch.float32),
            'cls': torch.tensor(labels, dtype=torch.int64)
        }
        
        return image, target

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    
    # EfficientDet expects all targets in the batch to be padded to the max number of boxes
    max_boxes = max([len(t['bbox']) for t in targets])
    max_boxes = max(max_boxes, 1) # Ensure at least 1 to prevent empty tensor crashes
    
    batch_bboxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    batch_cls = torch.zeros((len(batch), max_boxes), dtype=torch.int64) # 0 = background
    
    for i, t in enumerate(targets):
        num_boxes = len(t['bbox'])
        if num_boxes > 0:
            batch_bboxes[i, :num_boxes] = t['bbox']
            batch_cls[i, :num_boxes] = t['cls']
            
    return images, {'bbox': batch_bboxes, 'cls': batch_cls}

# ── Training loop ─────────────────────────────────────────────────────────────
def train_efficientdet():
    set_seeds()
    print("\n--- Training EfficientDet-D2 ---")
    print(f"[EfficientDet] Using device: {DEVICE}")

    # ── Hyperparams ───────────────────────────────────────────────────────────
    EPOCHS      = 100
    BATCH_SIZE  = 2       
    NUM_WORKERS = 0       
    LR          = 0.0001
    WEIGHT_DECAY= 0.00001
    CHECKPOINT  = 'checkpoints/efficientdet_best.pt'
    LAST_CKPT   = 'checkpoints/efficientdet_last.pt'

    save_hyperparams('efficientdet', {
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'lr': LR, 'weight_decay': WEIGHT_DECAY,
        'image_size': 512, 'amp': False, 'device': str(DEVICE)
    })

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[EfficientDet] Initializing model architecture...")
    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = 4
    config.image_size = (512, 512) 

    net = EfficientDet(config, pretrained_backbone=True)
    # DetBenchTrain wraps the model with the loss functions built-in
    model = DetBenchTrain(net, config)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch = 0
    best_loss   = float('inf')

    if os.path.exists(LAST_CKPT):
        print(f"[EfficientDet] Resuming from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        print(f"[EfficientDet] Resumed from epoch {start_epoch}")
    else:
        print("[EfficientDet] No checkpoint found, starting fresh.")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("[EfficientDet] Initializing dataloaders...")
    train_dataset = EffDetDataset(split='train')
    val_dataset   = EffDetDataset(split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False
    )

    # ── Epoch loop ────────────────────────────────────────────────────────────
    print("[EfficientDet] Starting training loop...")
    for epoch in range(start_epoch, EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = loss_dict['loss']

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.train() 
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                
                loss_dict = model(images, targets)
                val_loss += loss_dict['loss'].item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:>3}/{EPOCHS}  "
              f"train_loss: {avg_train_loss:.4f}  "
              f"val_loss: {avg_val_loss:.4f}  "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # ── Save last checkpoint (resume support) ─────────────────────────────
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'best_loss':  best_loss,
        }, LAST_CKPT)

        # ── Save best checkpoint ──────────────────────────────────────────────
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Extract just the backbone/head state dict without the DetBenchTrain loss wrapper
            torch.save(net.state_dict(), CHECKPOINT)
            print(f"  → New best saved (val_loss: {best_loss:.4f})")

    print(f"[EfficientDet] Training complete. Best checkpoint: {CHECKPOINT}")

if __name__ == '__main__':
    setup_environment()
    train_efficientdet()