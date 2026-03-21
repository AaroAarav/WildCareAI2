import os
import shutil
from ultralytics import YOLO
from ultralytics import settings
from utils import set_seeds, save_hyperparams, setup_environment, DEVICE

# Force YOLO to always save directly into checkpoints/, never into runs/detect/
settings.update({'runs_dir': os.path.abspath('checkpoints')})

def train_yolo(model_name, weights):
    set_seeds()
    print(f"\n--- Training {model_name.upper()} ---")

    project_dir = os.path.abspath('checkpoints')

    params = {
        'epochs': 100,
        'imgsz': 640,
        'batch': 4,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'cos_lr': True,
        'warmup_epochs': 3,
        'seed': 42,
        'mosaic': 1.0,
        'mixup': 0.1,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'device': 0,
        'amp': True,
        'cache': False,
        'workers': 2,
    }
    save_hyperparams(model_name, params)

    best_weights = os.path.join(project_dir, model_name, 'weights', 'best.pt')
    last_weights = os.path.join(project_dir, model_name, 'weights', 'last.pt')
    dest = os.path.join(project_dir, f'{model_name}_best.pt')

    # Check if training already fully completed
    if os.path.exists(best_weights):
        print(f"[{model_name}] Training already complete. Skipping to checkpoint copy.")
    elif os.path.exists(last_weights):
        # Partially trained — resume
        print(f"[{model_name}] Resuming from {last_weights}")
        model = YOLO(last_weights)
        model.train(resume=True)
    else:
        # Fresh start
        print(f"[{model_name}] No checkpoint found, starting fresh.")
        model = YOLO(weights)
        model.train(
            data='dataset.yaml',
            project=project_dir,
            name=model_name,
            exist_ok=True,
            **params
        )

    # Copy best.pt regardless of which path was taken
    if os.path.exists(best_weights):
        shutil.copy(best_weights, dest)
        print(f"[{model_name}] best.pt copied to {dest}")
    else:
        weights_dir = os.path.join(project_dir, model_name, 'weights')
        print(f"[{model_name}] WARNING: best.pt not found.")
        if os.path.exists(weights_dir):
            print(f"[{model_name}] Files in weights dir: {os.listdir(weights_dir)}")

if __name__ == '__main__':
    setup_environment()
    train_yolo('yolov8m', 'yolov8m.pt')
    train_yolo('yolov8n', 'yolov8n.pt')