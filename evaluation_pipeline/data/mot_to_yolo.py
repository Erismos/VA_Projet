import os
import cv2
from tqdm import tqdm

def convert_mot_to_yolo(mot_root, output_dir):
    """
    Converts MOT17 annotations to YOLO format.
    mot_root: Path to MOT17 folder (containing train/test)
    output_dir: Path to save YOLO formatted labels
    """
    train_path = os.path.join(mot_root, 'train')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    sequences = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    for seq in sequences:
        print(f"Processing sequence: {seq}")
        seq_path = os.path.join(train_path, seq)
        anno_file = os.path.join(seq_path, 'gt', 'gt.txt')
        img_dir = os.path.join(seq_path, 'img1')
        
        # Output label dir for this sequence
        seq_label_dir = os.path.join(output_dir, seq, 'labels')
        os.makedirs(seq_label_dir, exist_ok=True)
        
        # Get image dimensions from first image
        first_img = os.listdir(img_dir)[0]
        img = cv2.imread(os.path.join(img_dir, first_img))
        h, w, _ = img.shape
        
        with open(anno_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split(',')
            frame_idx = int(parts[0])
            # parts[1] is ID, we don't need it for detection
            x, y, width, height = map(float, parts[2:6])
            # class_id: MOT17 has multiple classes, usually 1 is pedestrian
            # We filter for class 1 if we only want pedestrians
            cls_id = int(parts[7])
            visibility = float(parts[8])
            
            if cls_id != 1 or visibility < 0.25:
                continue
            
            # YOLO format: cls x_center y_center width height (normalized)
            x_center = (x + width / 2) / w
            y_center = (y + height / 2) / h
            nw = width / w
            nh = height / h
            
            label_file = os.path.join(seq_label_dir, f"{frame_idx:06d}.txt")
            with open(label_file, 'a') as lf:
                lf.write(f"0 {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n")

if __name__ == "__main__":
    # Example usage
    # convert_mot_to_yolo('data/raw/MOT17', 'data/processed/MOT17_YOLO')
    pass
