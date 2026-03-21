import os
import glob
from pathlib import Path
from ultralytics.utils.downloads import download
import xml.etree.ElementTree as ET

def main():
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / 'step1_inspection_report.txt'
    
    print("Downloading African Wildlife dataset...")
    download('https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip', dir=Path('datasets'))
    
    aw_dir = Path('datasets/african-wildlife')
    kaggle_dir = Path('raw/kaggle_animals')
    
    report = []
    report.append("=== DATASET INSPECTION REPORT ===\n")
    
    # Inspect African Wildlife (YOLO Format)
    aw_images = list(aw_dir.rglob('*.jpg'))
    aw_labels = list(aw_dir.rglob('*.txt'))
    report.append(f"African Wildlife Images: {len(aw_images)}")
    report.append(f"African Wildlife Labels: {len(aw_labels)}")
    report.append("Format Detected: YOLO (.txt)\n")
    
    # Inspect Kaggle Dataset (Pascal VOC)
    if not kaggle_dir.exists():
        report.append("WARNING: Kaggle directory 'raw/kaggle_animals' not found. Please ensure it is unzipped.")
    else:
        k_images = list(kaggle_dir.rglob('images/*.jpg'))
        k_labels = list(kaggle_dir.rglob('annotations/*.xml'))
        report.append(f"Kaggle Dataset Images: {len(k_images)}")
        report.append(f"Kaggle Dataset Labels: {len(k_labels)}")
        report.append("Format Detected: Pascal VOC (.xml)\n")
        
        # basic missing check
        k_img_stems = {img.stem for img in k_images}
        k_lbl_stems = {lbl.stem for lbl in k_labels}
        missing_labels = k_img_stems - k_lbl_stems
        missing_images = k_lbl_stems - k_img_stems
        report.append(f"Kaggle Images missing labels: {len(missing_labels)}")
        report.append(f"Kaggle Labels missing images: {len(missing_images)}\n")

    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Inspection complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()