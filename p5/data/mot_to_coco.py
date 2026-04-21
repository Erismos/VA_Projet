import json
import os
import cv2

def mot_to_coco(mot_root, output_json, split_seqs):
    """
    Converts MOT17 annotations to COCO JSON format.
    mot_root: Root MOT17 folder
    output_json: Path to save COCO JSON
    split_seqs: List of sequence names to include
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
        "info": {"description": "MOT17 COCO Format"}
    }
    
    ann_id = 0
    img_id = 0
    
    for seq in split_seqs:
        seq_path = os.path.join(mot_root, 'train', seq)
        img_dir = os.path.join(seq_path, 'img1')
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        
        # Get image dimensions from first image
        first_img_name = sorted(os.listdir(img_dir))[0]
        img = cv2.imread(os.path.join(img_dir, first_img_name))
        h, w, _ = img.shape
        
        # Map frames to images
        images_in_seq = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        frame_to_img_id = {}
        for img_name in images_in_seq:
            frame_idx = int(img_name.split('.')[0])
            img_id += 1
            coco_data["images"].append({
                "id": img_id,
                "file_name": f"{seq}/img1/{img_name}",
                "width": w,
                "height": h
            })
            frame_to_img_id[frame_idx] = img_id
            
        # Add annotations
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_idx = int(parts[0])
                if frame_idx not in frame_to_img_id:
                    continue
                
                # Filter for pedestrians (class 1)
                cls_id = int(parts[7])
                visibility = float(parts[8])
                if cls_id != 1 or visibility < 0.25:
                    continue
                
                ann_id += 1
                x, y, width, height = map(float, parts[2:6])
                
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": frame_to_img_id[frame_idx],
                    "category_id": 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
    print(f"COCO JSON saved to {output_json}")

if __name__ == "__main__":
    pass
