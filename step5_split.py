import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

def main():
    cleaned_dir = Path('cleaned')
    dataset_dir = Path('dataset')
    
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect valid images
    images, primary_labels = [], []
    for txt_file in cleaned_dir.rglob('labels/*.txt'):
        img_file = txt_file.parent.parent / 'images' / f"{txt_file.stem}.jpg"
        if img_file.exists():
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Assign primary class for stratification approximation
                    first_class = int(lines[0].split()[0])
                    images.append((img_file, txt_file))
                    primary_labels.append(first_class)

    # 80 / 10 / 10 Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, primary_labels, test_size=0.10, stratify=primary_labels, random_state=42
    )
    
    X_train, X_val, _, _ = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=42
    ) # 0.1111 of 0.90 is ~0.10 of total

    splits = {'train': X_train, 'val': X_val, 'test': X_test}

    for split_name, files in splits.items():
        for img_path, txt_path in tqdm(files, desc=f"Copying {split_name} data"):
            shutil.copy(img_path, dataset_dir / 'images' / split_name / img_path.name)
            shutil.copy(txt_path, dataset_dir / 'labels' / split_name / txt_path.name)

    print("Splitting complete.")

if __name__ == "__main__":
    main()