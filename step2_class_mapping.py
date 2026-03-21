import json
from pathlib import Path
from collections import defaultdict

MASTER_CLASSES = {
    0: 'buffalo', 1: 'elephant', 2: 'rhino', 3: 'zebra',
    4: 'lion', 5: 'leopard', 6: 'giraffe', 7: 'hippopotamus',
    8: 'cheetah', 9: 'crocodile', 10: 'gorilla', 11: 'bear',
    12: 'wolf', 13: 'tiger', 14: 'fox', 15: 'deer',
    16: 'horse', 17: 'sheep', 18: 'cow', 19: 'dog'
}

SYNONYMS = {
    'rhinos': 'rhino', 'african elephant': 'elephant', 'hippo': 'hippopotamus',
    'buffaloes': 'buffalo', 'zebras': 'zebra', 'brown bear': 'bear', 
    'polar bear': 'bear', 'bull': 'cow', 'cattle': 'cow'
}

def main():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    kaggle_dir = Path('raw/kaggle_animals')
    
    class_counts = defaultdict(int)
    
    # Vacuum up ALL .txt files anywhere in the Kaggle directory
    if kaggle_dir.exists():
        for txt_file in kaggle_dir.rglob('*.txt'):
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Grabs the class name (even if multi-word like "Brown bear")
                            name = " ".join(parts[:-4]).lower().strip()
                            # If the label is just a number (YOLO id), skip mapping it here
                            if name.isdigit(): continue
                            
                            canonical_name = SYNONYMS.get(name, name)
                            class_counts[canonical_name] += 1
            except Exception:
                continue

    name_to_id = {v: k for k, v in MASTER_CLASSES.items()}
    final_mapping = {}
    dropped_classes = []
    
    for cls_name, count in class_counts.items():
        if count >= 150 and cls_name in name_to_id:
            final_mapping[cls_name] = name_to_id[cls_name]
        else:
            dropped_classes.append(f"{cls_name} (Count: {count})")
            
    for k, v in MASTER_CLASSES.items():
        if v not in final_mapping:
            final_mapping[v] = k

    with open(log_dir / 'class_map.json', 'w') as f:
        json.dump(final_mapping, f, indent=4)
        
    with open(log_dir / 'dropped_classes.txt', 'w') as f:
        f.write('\n'.join(dropped_classes))
        
    print("Class mapping complete. Mappings and drops saved to logs/")

if __name__ == "__main__":
    main()