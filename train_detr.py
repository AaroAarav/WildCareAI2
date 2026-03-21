import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
from pycocotools.coco import COCO

from utils import set_seeds, save_hyperparams, setup_environment, DEVICE

# ── Initialize Processor Globally ─────────────────────────────────────────────
# Do this ONCE so we don't spam Hugging Face's servers every batch!
PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# ── Dataset ───────────────────────────────────────────────────────────────────
class DetrDataset(Dataset):
    def __init__(self, split):
        self.root = f"dataset/images/{split}"
        self.coco = COCO(f"coco_annotations/instances_{split}.json")
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        # Load Image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert("RGB")

        # Parse Annotations
        annotations = []
        for ann in coco_annotation:
            # Skip invalid boxes
            if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
                continue
                
            annotations.append({
                "area": ann['area'],
                "iscrowd": ann.get('iscrowd', 0),
                "image_id": img_id,
                "bbox": ann['bbox'],  # COCO format [x_min, y_min, w, h]
                "category_id": ann['category_id']  # Keep exactly 0, 1, 2, 3
            })
            
        target = {'image_id': img_id, 'annotations': annotations}
        return image, target

def collate_fn(batch):
    # The processor dynamically pads images to the largest size in the batch
    # and normalizes bounding boxes specifically for DETR
    pixel_values = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Use the globally initialized processor
    encoding = PROCESSOR(images=pixel_values, annotations=targets, return_tensors="pt")
    return encoding

# ── Training loop ─────────────────────────────────────────────────────────────
def train_detr():
    set_seeds()
    print("\n--- Training DETR ResNet-50 ---")
    print(f"[DETR] Using device: {DEVICE}")

    # ── Hyperparams ───────────────────────────────────────────────────────────
    EPOCHS      = 100
    BATCH_SIZE  = 2       # Safe for 6GB VRAM
    NUM_WORKERS = 0       # Must be 0 on Windows
    STEP_SIZE   = 70
    CHECKPOINT  = 'checkpoints/detr_best.pt'
    LAST_CKPT   = 'checkpoints/detr_last.pt'

    save_hyperparams('detr', {
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'step_size': STEP_SIZE, 'amp': False, 'device': str(DEVICE)
    })

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[DETR] Initializing model architecture...")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # DETR requires specific learning rates for different parts of the network
    optimizer = torch.optim.AdamW([
        {'params': model.model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.model.encoder.parameters(),  'lr': 1e-4},
        {'params': model.model.decoder.parameters(),  'lr': 1e-4},
        {'params': model.class_labels_classifier.parameters(), 'lr': 1e-4},
        {'params': model.bbox_predictor.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch = 0
    best_loss   = float('inf')

    if os.path.exists(LAST_CKPT):
        print(f"[DETR] Resuming from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        print(f"[DETR] Resumed from epoch {start_epoch}")
    else:
        print("[DETR] No checkpoint found, starting fresh.")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("[DETR] Initializing dataloaders...")
    train_dataset = DetrDataset(split='train')
    val_dataset   = DetrDataset(split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False
    )

    # ── Epoch loop ────────────────────────────────────────────────────────────
    print("[DETR] Starting training loop...")
    for epoch in range(start_epoch, EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move HuggingFace encoded batch to GPU
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask = batch["pixel_mask"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            optimizer.zero_grad()

            # Forward pass (DETR automatically calculates the Hungarian matching loss if labels are provided)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            
            # Gradient clipping is highly recommended for DETR
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval() 
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                pixel_mask = batch["pixel_mask"].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                val_loss += outputs.loss.item()

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
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  → New best saved (val_loss: {best_loss:.4f})")

    print(f"[DETR] Training complete. Best checkpoint: {CHECKPOINT}")

if __name__ == '__main__':
    setup_environment()
    train_detr()