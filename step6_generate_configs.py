import json
import cv2
from pathlib import Path
from tqdm import tqdm

def main():
    dataset_dir = Path('dataset')
    coco_dir = Path('coco_annotations')
    coco_dir.mkdir(exist_ok=True)
    
    # Load Master Classes
    with open('logs/class_map.json', 'r') as f:
        class_map_raw = json.load(f)
    # Deduplicate values to build true MASTER mapping
    master_classes = {}
    for k, v in class_map_raw.items():
        if isinstance(k, str) and not k.isdigit():
            master_classes[v] = k # keeps overriding but standardizes
            
    # Hardcode overwrite to ensure clean names for standard 20
    master_names = {
    0: 'buffalo', 1: 'elephant', 2: 'rhino', 3: 'zebra'
}

    # Generate dataset.yaml
    yaml_content = f"""path: ./dataset
train: images/train
val: images/val
test: images/test
nc: {len(master_names)}
names:
"""
    for idx in sorted(master_names.keys()):
        yaml_content += f"  {idx}: {master_names[idx]}\n"
        
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)

    # Generate COCO JSON
    categories = [{"id": k, "name": v} for k, v in master_names.items()]
    
    for split in ['train', 'val', 'test']:
        coco_format = {
            "info": {"description": "Unified Wildlife Dataset"},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        img_id = 0
        ann_id = 0
        
        img_dir = dataset_dir / 'images' / split
        lbl_dir = dataset_dir / 'labels' / split
        
        for img_path in tqdm(list(img_dir.glob('*.jpg')), desc=f"COCO {split}"):
            img = cv2.imread(str(img_path))
            h, w_img, _ = img.shape
            
            coco_format["images"].append({
                "id": img_id, "file_name": img_path.name, "width": w_img, "height": h
            })
            
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        c, x_c, y_c, w_norm, h_norm = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        
                        w_px = w_norm * w_img
                        h_px = h_norm * h
                        x_tl = (x_c - w_norm/2) * w_img
                        y_tl = (y_c - h_norm/2) * h
                        area = w_px * h_px
                        
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": c,
                            "bbox": [x_tl, y_tl, w_px, h_px],
                            "area": area,
                            "iscrowd": 0
                        })
                        ann_id += 1
            img_id += 1
            
        with open(coco_dir / f'instances_{split}.json', 'w') as f:
            json.dump(coco_format, f)

    print("Config generation complete.")

if __name__ == "__main__":
    main()