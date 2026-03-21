import os
import sys
import yaml
from glob import glob
from pycocotools.coco import COCO

MASTER_CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def verify_dataset():
    print("Starting pre-training dataset verification for 4 classes...\n")
    
    if not os.path.exists('dataset.yaml'):
        fail("dataset.yaml not found.")
    
    with open('dataset.yaml', 'r') as f:
        ds_yaml = yaml.safe_load(f)
    
    if ds_yaml.get('nc') != 4:
        fail(f"dataset.yaml nc != 4. Found: {ds_yaml.get('nc')}")
        
    yaml_names = ds_yaml.get('names', {})
    if isinstance(yaml_names, dict):
        yaml_names = [yaml_names[k] for k in sorted(yaml_names.keys())]
        
    if yaml_names != MASTER_CLASSES:
        fail(f"dataset.yaml class names mismatch. Expected: {MASTER_CLASSES}")

    splits = ['train', 'val', 'test']
    summary = []

    for split in splits:
        print(f"Verifying {split} split...")
        
        # YOLO Verification
        img_dir = f'dataset/images/{split}'
        lbl_dir = f'dataset/labels/{split}'
        
        yolo_images = glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png'))
        yolo_img_count = len(yolo_images)
        yolo_ann_count = 0
        yolo_classes_present = set()
        
        for img_path in yolo_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            if not os.path.exists(lbl_path):
                fail(f"Missing YOLO label for image: {img_path}")
                
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    fail(f"Empty YOLO label file: {lbl_path}")
                
                yolo_ann_count += len(lines)
                for line in lines:
                    cls_id = int(line.split()[0])
                    if not (0 <= cls_id <= 3):
                        fail(f"Invalid class ID {cls_id} (must be 0-3) in {lbl_path}")
                    yolo_classes_present.add(cls_id)

        # COCO Verification
        coco_json_path = f'coco_annotations/instances_{split}.json'
        if not os.path.exists(coco_json_path):
            fail(f"Missing COCO JSON: {coco_json_path}")
            
        try:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            coco = COCO(coco_json_path)
            sys.stdout = old_stdout
        except Exception as e:
            fail(f"Failed to load COCO JSON {coco_json_path}: {e}")

        coco_img_ids = coco.getImgIds()
        coco_img_count = len(coco_img_ids)
        coco_ann_count = len(coco.getAnnIds())
        
        coco_cats = coco.loadCats(coco.getCatIds())
        for cat in coco_cats:
            if cat['id'] < 0 or cat['id'] > 3:
                fail(f"COCO Category ID out of bounds [0-3]: {cat['id']}")
            if cat['name'] != MASTER_CLASSES[cat['id']]:
                fail(f"COCO Category name mismatch for ID {cat['id']}")

        if yolo_img_count != coco_img_count:
            fail(f"YOLO/COCO image count mismatch in {split}: YOLO={yolo_img_count}, COCO={coco_img_count}")

        summary.append({
            'Split': split,
            'YOLO Images': yolo_img_count,
            'COCO Images': coco_img_count,
            'Annotations': coco_ann_count,
            'Classes Present': len(yolo_classes_present)
        })

    print("\nDataset Verification Successful! Summary:")
    print("-" * 75)
    print(f"{'Split':<10} | {'YOLO Images':<15} | {'COCO Images':<15} | {'Annotations':<15} | {'Classes Present':<15}")
    print("-" * 75)
    for row in summary:
        print(f"{row['Split']:<10} | {row['YOLO Images']:<15} | {row['COCO Images']:<15} | {row['Annotations']:<15} | {row['Classes Present']:<15}")
    print("-" * 75)

if __name__ == '__main__':
    verify_dataset()