import json
from pathlib import Path
from pycocotools.coco import COCO

def main():
    dataset_dir = Path('dataset')
    log_dir = Path('logs')
    
    report_html = "<html><body><h1>Dataset Generation Report</h1>"
    report_html += "<table border='1'><tr><th>Split</th><th>Total Images</th><th>Valid COCO Check</th></tr>"
    
    for split in ['train', 'val', 'test']:
        img_count = len(list((dataset_dir / 'images' / split).glob('*.jpg')))
        coco_path = f'coco_annotations/instances_{split}.json'
        
        try:
            coco = COCO(coco_path)
            status = "PASS"
        except Exception as e:
            status = f"FAIL ({e})"
            
        report_html += f"<tr><td>{split}</td><td>{img_count}</td><td>{status}</td></tr>"

    report_html += "</table></body></html>"
    
    with open(log_dir / 'dataset_report.html', 'w') as f:
        f.write(report_html)
        
    print("Validation complete. Report generated at logs/dataset_report.html")

if __name__ == "__main__":
    main()