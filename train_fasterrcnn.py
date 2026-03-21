import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import torchvision.transforms.functional as F
from pycocotools.coco import COCO

from utils import set_seeds, save_hyperparams, setup_environment, DEVICE

# ── Dataset ───────────────────────────────────────────────────────────────────
class AnimalDataset(Dataset):
    def __init__(self, split):
        self.root = f"dataset/images/{split}"
        self.coco = COCO(f"coco_annotations/instances_{split}.json")
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        # Load Image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = F.to_tensor(img) # Converts to FloatTensor[C, H, W] in range [0, 1]

        # Parse Annotations
        num_objs = len(coco_annotation)
        boxes = []
        labels = []
        
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            width = coco_annotation[i]['bbox'][2]
            height = coco_annotation[i]['bbox'][3]
            
            # Safety check: skip invalid boxes that might crash CUDA
            if width <= 0 or height <= 0:
                continue
                
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(coco_annotation[i]['category_id'] + 1)

        # Handle empty images gracefully
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ── Training loop ─────────────────────────────────────────────────────────────
def train_faster_rcnn():
    set_seeds()
    print("\n--- Training Faster R-CNN ---")
    print(f"[Faster R-CNN] Using device: {DEVICE}")

    # ── Hyperparams ───────────────────────────────────────────────────────────
    EPOCHS      = 100
    BATCH_SIZE  = 2       # REDUCED: Safe for 6GB VRAM in standard FP32
    LR          = 0.005
    MOMENTUM    = 0.9
    WEIGHT_DECAY= 0.0005
    STEP_SIZE   = 30
    GAMMA       = 0.1
    NUM_WORKERS = 0       # MUST BE 0 FOR WINDOWS
    CHECKPOINT  = 'checkpoints/fasterrcnn_best.pt'
    LAST_CKPT   = 'checkpoints/fasterrcnn_last.pt'

    save_hyperparams('fasterrcnn', {
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'lr': LR, 'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY,
        'step_size': STEP_SIZE, 'amp': False, 'device': str(DEVICE)
    })

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[Faster R-CNN] Initializing model architecture...")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)  # 4 animals + bg

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    
    print("[Faster R-CNN] Moving model to GPU...")
    model.to(DEVICE)

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch = 0
    best_loss   = float('inf')

    if os.path.exists(LAST_CKPT):
        print(f"[Faster R-CNN] Resuming from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        print(f"[Faster R-CNN] Resumed from epoch {start_epoch}")
    else:
        print("[Faster R-CNN] No checkpoint found, starting fresh.")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("[Faster R-CNN] Initializing dataloaders...")
    train_dataset = AnimalDataset(split='train')
    val_dataset   = AnimalDataset(split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False
    )

    # ── Epoch loop ────────────────────────────────────────────────────────────
    print("[Faster R-CNN] Starting training loop...")
    for epoch in range(start_epoch, EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Stripped Autocast - standard FP32 forward pass
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Print an update every 50 batches so you know it's not frozen
            if (batch_idx + 1) % 50 == 0:
                print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.train()  # Faster R-CNN needs train mode to compute val loss
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images  = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()

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

    print(f"[Faster R-CNN] Training complete. Best checkpoint: {CHECKPOINT}")

if __name__ == '__main__':
    setup_environment()
    train_faster_rcnn()