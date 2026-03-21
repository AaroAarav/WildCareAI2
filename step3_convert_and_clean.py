import cv2
import json
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm

SYNONYMS = {
    'rhinos': 'rhino', 'african elephant': 'elephant', 'hippo': 'hippopotamus',
    'buffaloes': 'buffalo', 'zebras': 'zebra', 'brown bear': 'bear', 
    'polar bear': 'bear', 'bull': 'cow', 'cattle': 'cow'
}

def get_image_hash(filepath):
    try:
        return str(imagehash.phash(Image.open(filepath)))
    except:
        return None

def main():
    log_dir = Path('logs')
    cleaned_dir = Path('cleaned')
    
    # Force create directories immediately
    log_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True) 

    # Load Class Map
    with open(log_dir / 'class_map.json', 'r') as f:
        class_map = json.load(f)

    seen_hashes = set()
    cleaning_log = []

    def log_drop(reason, filepath):
        cleaning_log.append(f"{reason}: {filepath}")

    def process_dataset(dataset_dir, dataset_type):
        all_files = list(Path(dataset_dir).rglob('*'))
        image_paths = [p for p in all_files if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"\n{'='*40}")
        print(f"🔎 INSPECTING: {dataset_type}")
        print(f"   Found {len(image_paths)} image files in {dataset_dir}")
        print(f"{'='*40}")
        
        if len(image_paths) == 0:
            print(f"❌ WARNING: No images found!")
            return

        valid_images_saved = 0

        for img_path in tqdm(image_paths, desc=f"Processing {dataset_type}"):
            img = cv2.imread(str(img_path))
            if img is None:
                log_drop("Unreadable Image", img_path)
                continue
            
            h, w_img, _ = img.shape
            
            img_hash = get_image_hash(img_path)
            if img_hash and img_hash in seen_hashes:
                log_drop("Duplicate Image", img_path)
                continue
            if img_hash:
                seen_hashes.add(img_hash)

            valid_boxes = []
            
            # Find matching text file safely
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists() and 'images' in img_path.parts:
                parts = list(img_path.parts)
                parts[parts.index('images')] = 'labels' 
                txt_path = Path(*parts).with_suffix('.txt')

            if txt_path.exists():
                try:
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            
                            # KAGGLE DATASET PARSING
                            if dataset_type == 'Kaggle' and len(parts) >= 5:
                                cls_name = " ".join(parts[:-4]).lower().strip()
                                canonical_name = SYNONYMS.get(cls_name, cls_name)
                                
                                if canonical_name in class_map:
                                    c_id = class_map[canonical_name]
                                    xmin, ymin = float(parts[-4]), float(parts[-3])
                                    xmax, ymax = float(parts[-2]), float(parts[-1])
                                    
                                    # Math: Convert Absolute Pixels -> YOLO Normalized
                                    x_c = (xmin + xmax) / (2 * w_img)
                                    y_c = (ymin + ymax) / (2 * h)
                                    w_norm = (xmax - xmin) / w_img
                                    h_norm = (ymax - ymin) / h
                                    
                                    valid_boxes.append((c_id, x_c, y_c, w_norm, h_norm))
                                    
                            # AFRICAN WILDLIFE PARSING
                            elif dataset_type == 'AfricanWildlife' and len(parts) >= 5:
                                try:
                                    c_id = int(parts[0])
                                    x_c, y_c = float(parts[1]), float(parts[2])
                                    w_norm, h_norm = float(parts[3]), float(parts[4])
                                    valid_boxes.append((c_id, x_c, y_c, w_norm, h_norm))
                                except ValueError:
                                    pass
                except Exception as e:
                    log_drop(f"TXT Parse Error ({e})", txt_path)
            else:
                log_drop("No matching TXT label found", img_path)

            # Filter bad boxes
            cleaned_boxes = []
            for box in valid_boxes:
                c, x, y, bw, bh = box
                if 0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1 and (bw * bh) >= 0.0005:
                    cleaned_boxes.append(box)

            if not cleaned_boxes:
                log_drop("No valid boxes", img_path)
                continue
            cleaned_boxes = [box for box in cleaned_boxes if box[0] in [0, 1, 2, 3]]
            # Save to Cleaned Directory
            out_img_dir = cleaned_dir / dataset_type / 'images'
            out_lbl_dir = cleaned_dir / dataset_type / 'labels'
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            new_img_path = out_img_dir / img_path.name
            if not new_img_path.exists():
                cv2.imwrite(str(new_img_path), img)
                
            with open(out_lbl_dir / f"{img_path.stem}.txt", 'w') as f:
                for box in cleaned_boxes:
                    f.write(f"{box[0]} {' '.join(f'{v:.6f}' for v in box[1:])}\n")
                    
            valid_images_saved += 1

        print(f"✅ Successfully cleaned and saved {valid_images_saved} images for {dataset_type}.")

    # Run for both datasets
    process_dataset('datasets/african-wildlife', 'AfricanWildlife')
    process_dataset('raw/kaggle_animals', 'Kaggle')

    with open(log_dir / 'step3_cleaning_log.txt', 'w') as f:
        f.write('\n'.join(cleaning_log))

if __name__ == "__main__":
    main()