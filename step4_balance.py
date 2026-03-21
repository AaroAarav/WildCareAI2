import cv2
import random
from pathlib import Path
from collections import defaultdict
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    cleaned_dir = Path('cleaned')
    log_dir = Path('logs')
    
    # Gather distribution
    image_to_classes = defaultdict(list)
    class_counts = defaultdict(int)
    
    for txt_file in cleaned_dir.rglob('labels/*.txt'):
        with open(txt_file, 'r') as f:
            classes = [int(line.split()[0]) for line in f]
            if classes:
                image_to_classes[txt_file] = classes
                for c in set(classes): # Count image presence, not raw boxes
                    class_counts[c] += 1

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.Rotate(limit=15, p=0.4),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=1024, min_visibility=0.3))

    undersampled_log = []
    random.seed(42)

    # Note: Complex multi-label datasets require iterative balancing. 
    # This simplified version targets the primary class of an image.
    for cls_id, count in class_counts.items():
        cls_images = [k for k, v in image_to_classes.items() if cls_id in v]
        
        # UNDERSAMPLE
        if count > 3000:
            to_remove = random.sample(cls_images, count - 3000)
            for path in to_remove:
                undersampled_log.append(str(path))
                # Remove from tracking (actual file deletion is optional, 
                # but we'll remove it from the pipeline pool here)
                if path in image_to_classes:
                    del image_to_classes[path]
                    
        # OVERSAMPLE
        elif count < 300:
            deficit = 300 - count
            if not cls_images: continue
            
            for i in tqdm(range(deficit), desc=f"Augmenting Class {cls_id}"):
                src_txt = random.choice(cls_images)
                src_img = src_txt.parent.parent / 'images' / f"{src_txt.stem}.jpg"
                
                img = cv2.imread(str(src_img))
                bboxes, labels = [], []
                with open(src_txt, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])
                
                try:
                    transformed = aug(image=img, bboxes=bboxes, class_labels=labels)
                    if not transformed['bboxes']: continue
                    
                    new_stem = f"{src_img.stem}_aug{i:03d}"
                    new_img_path = src_img.parent / f"{new_stem}.jpg"
                    new_txt_path = src_txt.parent / f"{new_stem}.txt"
                    
                    cv2.imwrite(str(new_img_path), transformed['image'])
                    with open(new_txt_path, 'w') as f:
                        for box, lbl in zip(transformed['bboxes'], transformed['class_labels']):
                            f.write(f"{lbl} {' '.join(f'{v:.6f}' for v in box)}\n")
                            
                    image_to_classes[new_txt_path] = transformed['class_labels']
                except Exception as e:
                    # Albumentations can occasionally fail on tight bounding boxes during rotation
                    continue

    with open(log_dir / 'undersampled.txt', 'w') as f:
        f.write('\n'.join(undersampled_log))
        
    # Plot final distribution
    final_counts = defaultdict(int)
    for classes in image_to_classes.values():
        for c in set(classes):
            final_counts[c] += 1
            
    plt.bar(final_counts.keys(), final_counts.values())
    plt.title("Class Distribution After Balancing")
    plt.xlabel("Class ID")
    plt.ylabel("Images")
    plt.savefig(log_dir / 'step4_class_distribution.png')
    print("Balancing complete.")

if __name__ == "__main__":
    main()