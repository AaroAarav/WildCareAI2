import os
import json
import random
import numpy as np
import torch
import shutil
from ultralytics import YOLO

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from effdet import create_model, get_efficientdet_config, EfficientDet, DetBenchTrain
from transformers import DetrForObjectDetection

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # keep False for reproducibility

def save_hyperparams(model_name, params):
    os.makedirs('results', exist_ok=True)
    with open(f'results/hyperparams_{model_name}.json', 'w') as f:
        json.dump(params, f, indent=4)

def setup_environment():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    with open('results/env_info.txt', 'w') as f:
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available:  {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Model:       {torch.cuda.get_device_name(0)}\n")
            f.write(f"VRAM (MB):       {torch.cuda.get_device_properties(0).total_memory // 1024**2}\n")

    print(f"[Setup] Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")

# ── YOLO ──────────────────────────────────────────────────────────────────────
def train_yolo(model_name, weights):
    set_seeds()
    print(f"\n--- Training {model_name.upper()} ---")

    params = {
        # Core
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,          # RTX 4050 safe batch size (was 16 — drop to 8 for VRAM headroom)
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'cos_lr': True,
        'warmup_epochs': 3,
        'seed': 42,
        # Augmentation
        'mosaic': 1.0,
        'mixup': 0.1,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # GPU / speed
        'device': 0,         # explicitly use GPU 0
        'amp': True,         # mixed precision — major speed boost on RTX 40xx Ada
        'cache': 'ram',      # cache images in RAM for faster epoch iterations
        'workers': 4,        # safe worker count for Windows
    }
    save_hyperparams(model_name, params)

    model = YOLO(weights)
    model.train(
        data='dataset.yaml',
        project='checkpoints',
        name=model_name,
        exist_ok=True,
        **params
    )

    best_weights = f'checkpoints/{model_name}/weights/best.pt'
    shutil.copy(best_weights, f'checkpoints/{model_name}_best.pt')
    print(f"[{model_name}] Best weights saved to checkpoints/{model_name}_best.pt")

# ── Faster R-CNN ──────────────────────────────────────────────────────────────
def train_faster_rcnn():
    set_seeds()
    print("\n--- Training Faster R-CNN ---")

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 4 animals + 1 background = 5
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    model.to(DEVICE)

    # Enable mixed precision scaler for RTX 40xx
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    save_hyperparams('fasterrcnn', {
        'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005,
        'step_size': 30, 'amp': True, 'device': str(DEVICE)
    })

    print("[Pipeline stub] Faster R-CNN initialized for 4 classes. Saving dummy checkpoint.")
    torch.save(model.state_dict(), 'checkpoints/fasterrcnn_best.pt')

# ── EfficientDet ──────────────────────────────────────────────────────────────
def train_efficientdet():
    set_seeds()
    print("\n--- Training EfficientDet-D2 ---")

    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = 4
    # Reduced from 768 → 512 to fit RTX 4050 VRAM comfortably
    config.image_size = (512, 512)

    net = EfficientDet(config, pretrained_backbone=True)
    model = DetBenchTrain(net, config)
    model.to(DEVICE)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0001, weight_decay=0.00001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    save_hyperparams('efficientdet', {
        'lr': 0.0001, 'weight_decay': 0.00001, 'T_max': 100,
        'image_size': 512, 'amp': True, 'device': str(DEVICE)
    })

    print("[Pipeline stub] EfficientDet initialized for 4 classes. Saving dummy checkpoint.")
    torch.save(model.state_dict(), 'checkpoints/efficientdet_best.pt')

# ── DETR ──────────────────────────────────────────────────────────────────────
def train_detr():
    set_seeds()
    print("\n--- Training DETR ResNet-50 ---")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    # Mixed precision scaler — DETR is the heaviest model here
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    optimizer = torch.optim.AdamW([
        {'params': model.model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.model.encoder.parameters(),  'lr': 1e-4},
        {'params': model.model.decoder.parameters(),  'lr': 1e-4},
        {'params': model.class_labels_classifier.parameters(), 'lr': 1e-4},
        {'params': model.bbox_predictor.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)

    save_hyperparams('detr', {
        'backbone_lr': 1e-5, 'lr': 1e-4, 'weight_decay': 1e-4,
        'step_size': 70, 'amp': True, 'device': str(DEVICE),
        'note': 'Use batch_size=2 in dataloader to stay within 4GB VRAM'
    })

    print("[Pipeline stub] DETR initialized for 4 classes. Saving dummy checkpoint.")
    torch.save(model.state_dict(), 'checkpoints/detr_best.pt')

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    setup_environment()
    train_yolo('yolov8m', 'yolov8m.pt')
    train_yolo('yolov8n', 'yolov8n.pt')
    train_faster_rcnn()
    train_efficientdet()
    train_detr()