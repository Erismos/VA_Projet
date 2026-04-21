import os
import random
import shutil

def split_sequences(processed_dir, train_ratio=0.8):
    """
    Splits sequences into train and validation sets.
    """
    sequences = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    random.shuffle(sequences)
    
    split_idx = int(len(sequences) * train_ratio)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    return train_seqs, val_seqs

def create_yolo_dataset(processed_dir, output_dir, train_seqs, val_seqs):
    """
    Organizes files into a YOLO-ready directory structure.
    """
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
        target_seqs = train_seqs if split == 'train' else val_seqs
        
        for seq in target_seqs:
            img_dir = f"data/raw/MOT17/train/{seq}/img1"
            label_dir = os.path.join(processed_dir, seq, 'labels')
            
            for f in os.listdir(label_dir):
                if f.endswith('.txt'):
                    # Copy label
                    shutil.copy(os.path.join(label_dir, f), os.path.join(output_dir, 'labels', split, f"{seq}_{f}"))
                    # Copy image (need to match frame name)
                    img_name = f.replace('.txt', '.jpg')
                    shutil.copy(os.path.join(img_dir, img_name), os.path.join(output_dir, 'images', split, f"{seq}_{img_name}"))

if __name__ == "__main__":
    # Example usage logic
    pass
