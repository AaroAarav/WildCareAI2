import os
import json
import time
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# COCO tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Model Specific Imports
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

# Constants
MODELS = ['yolov8m', 'yolov8n', 'fasterrcnn', 'efficientdet', 'detr']
CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global dictionary to store dynamically calculated FPS
FPS_RECORDS = {}

def run_inference_and_save(model_name, coco_gt):
    pred_path = f'results/predictions_{model_name}.json'
    
    # If you need to re-run inference from scratch, delete the predictions folder/files manually
    if os.path.exists(pred_path):
        print(f"  [Info] Predictions for {model_name} already exist. Loading from file.")
        return pred_path

    print(f"  [Inference] Loading {model_name} and running on test set...")
    predictions = []
    img_ids = coco_gt.getImgIds()
    
    # --- MODEL INITIALIZATION ---
    if 'yolo' in model_name:
        model = YOLO(f'checkpoints/{model_name}_best.pt')
        
    elif model_name == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
        model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt', map_location=DEVICE))
        model.to(DEVICE).eval()
        
    elif model_name == 'detr':
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=4, ignore_mismatched_sizes=True)
        model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt', map_location=DEVICE))
        model.to(DEVICE).eval()
        
    elif model_name == 'efficientdet':
        config = get_efficientdet_config('tf_efficientdet_d2')
        config.num_classes = 4
        config.image_size = (512, 512)
        net = EfficientDet(config, pretrained_backbone=False)
        net.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pt', map_location=DEVICE))
        model = DetBenchPredict(net)
        model.to(DEVICE).eval()

    # --- INFERENCE LOOP ---
    start_time = time.time()
    
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join('dataset/images/test', img_info['file_name'])
        
        # 1. YOLOv8
        if 'yolo' in model_name:
            results = model.predict(img_path, verbose=False, device=DEVICE)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                if score > 0.05:
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(box.cls[0].item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(score, 4)
                    })

        # 2. Faster R-CNN
        elif model_name == 'fasterrcnn':
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_tensor)[0]
            for i in range(len(out['boxes'])):
                score = out['scores'][i].item()
                if score > 0.05:
                    x1, y1, x2, y2 = out['boxes'][i].tolist()
                    label = out['labels'][i].item() - 1 # Shift +1 background back to normal
                    predictions.append({
                        "image_id": img_id, "category_id": label, 
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score, 4)
                    })

        # 3. DETR
        elif model_name == 'detr':
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)[0]
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({
                    "image_id": img_id, "category_id": label.item(), 
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score.item(), 4)
                })

        # 4. EfficientDet
        elif model_name == 'efficientdet':
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            # Strict 512x512 Resize and Normalize for EfficientDet inference
            img_resized = cv2.resize(img, (512, 512))
            img_normalized = (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                img_info = {
                    "img_scale": torch.tensor([1.0], dtype=torch.float32).to(DEVICE),
                    "img_size": torch.tensor([[512, 512]], dtype=torch.float32).to(DEVICE)
                }
                out = model(img_tensor, img_info=img_info)
            
            dets = out[0] # [num_boxes, 6] -> [x1, y1, x2, y2, score, class]
            for det in dets:
                score = det[4].item()
                if score > 0.05:
                    x1, y1, x2, y2 = det[0:4].tolist()
                    # Rescale boxes back to original test image dimensions
                    x1 = x1 * (orig_w / 512.0)
                    y1 = y1 * (orig_h / 512.0)
                    x2 = x2 * (orig_w / 512.0)
                    y2 = y2 * (orig_h / 512.0)
                    
                    label = int(det[5].item()) - 1 # Shift +1 background back to normal
                    predictions.append({
                        "image_id": img_id, "category_id": label, 
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(score, 4)
                    })

    # --- SAVE PREDICTIONS & CALCULATE FPS ---
    total_time = time.time() - start_time
    FPS_RECORDS[model_name] = len(img_ids) / total_time
    
    os.makedirs('results', exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
        
    return pred_path

def evaluate_model(model_name, coco_gt):
    print(f"\nEvaluating {model_name}...")
    pred_path = run_inference_and_save(model_name, coco_gt)
    
    try:
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats = coco_eval.stats
        
        def safe_stat(idx):
            return stats[idx] if len(stats) > idx and stats[idx] != -1 else 0.0

        metrics = {
            'model': model_name,
            'AP_5_95': safe_stat(0), 'AP_50': safe_stat(1), 'AP_75': safe_stat(2),
            'AP_small': safe_stat(3), 'AP_medium': safe_stat(4), 'AP_large': safe_stat(5),
            'AR_100': safe_stat(8), 'FPS': FPS_RECORDS.get(model_name, 0.0)
        }
            
        return metrics, coco_eval
    except Exception as e:
        print(f"Skipping {model_name} evaluation due to error: {e}")
        return None, None

def plot_bar_chart(df):
    plt.figure(figsize=(10, 6), facecolor='#F7F9FC')
    ax = plt.gca()
    ax.set_facecolor('#F7F9FC')
    
    x = np.arange(len(df['model']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['AP_5_95'], width, label='AP@[.5:.95]', color='#1f77b4')
    bars2 = ax.bar(x + width/2, df['AP_50'], width, label='AP@.50', color='#ff7f0e')
    
    ax.set_ylabel('Score')
    ax.set_title('Animal Detection Model Comparison (mAP Scores)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'])
    ax.legend()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
                        
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/model_comparison_bar.png', dpi=180, bbox_inches='tight')
    plt.close()

def plot_heatmap(per_class_data):
    df = pd.DataFrame(per_class_data)
    if df.empty: return
    
    pivot = df.pivot(index="model", columns="class_name", values="AP@.50")
    pivot = pivot[CLASSES] # Ensure consistent order
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Per-Class AP@.50 Heatmap")
    plt.savefig('results/figures/per_class_heatmap.png', bbox_inches='tight')
    plt.close()

def main():
    print(f"Using device: {DEVICE}")
    coco_gt = COCO('coco_annotations/instances_test.json')
    all_metrics = []
    per_class_data = []

    for model_name in MODELS:
        metrics, coco_eval = evaluate_model(model_name, coco_gt)
        if metrics:
            all_metrics.append(metrics)
            
            coco_eval.params.iouThrs = np.array([0.5])
            # Loop strictly over 0 to 3
            for cat_id in range(4):
                coco_eval.params.catIds = [cat_id]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                ap50 = coco_eval.stats[1] if len(coco_eval.stats) > 1 and coco_eval.stats[1] != -1 else 0.0
                per_class_data.append({
                    'model': model_name,
                    'class_id': cat_id,
                    'class_name': CLASSES[cat_id],
                    'AP@.50': ap50 * 100
                })

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv('results/metrics_summary.csv', index=False)
        
        df_per_class = pd.DataFrame(per_class_data)
        df_per_class.to_csv('results/per_class_ap.csv', index=False)
        
        plot_bar_chart(df_metrics)
        plot_heatmap(per_class_data)
        print("\n========================================================")
        print("🎉 Evaluation Complete! All CSVs and Plots saved to results/")
        print("========================================================")

if __name__ == '__main__':
    main()