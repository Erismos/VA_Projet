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

            if not os.path.isdir(label_dir) or not os.path.isdir(img_dir):
                continue
            
            for f in os.listdir(label_dir):
                if f.endswith('.txt'):
                    # Copy label
                    src_label = os.path.join(label_dir, f)
                    dst_label = os.path.join(output_dir, 'labels', split, f"{seq}_{f}")
                    shutil.copy(src_label, dst_label)
                    # Copy image (need to match frame name)
                    img_name = f.replace('.txt', '.jpg')
                    src_img = os.path.join(img_dir, img_name)
                    dst_img = os.path.join(output_dir, 'images', split, f"{seq}_{img_name}")
                    if os.path.isfile(src_img):
                        shutil.copy(src_img, dst_img)


def validate_yolo_dataset(output_dir, train_seqs, val_seqs):
    """
    Validates that copied YOLO dataset has aligned image/label pairs and non-empty splits.
    """
    result = {
        "ok": True,
        "splits": {},
        "errors": [],
    }

    expected = {
        "train": set(train_seqs),
        "val": set(val_seqs),
    }

    for split in ["train", "val"]:
        images_dir = os.path.join(output_dir, "images", split)
        labels_dir = os.path.join(output_dir, "labels", split)

        img_files = [f for f in os.listdir(images_dir)] if os.path.isdir(images_dir) else []
        lbl_files = [f for f in os.listdir(labels_dir)] if os.path.isdir(labels_dir) else []

        img_stems = {os.path.splitext(f)[0] for f in img_files if f.lower().endswith(".jpg")}
        lbl_stems = {os.path.splitext(f)[0] for f in lbl_files if f.lower().endswith(".txt")}

        missing_images = sorted(lbl_stems - img_stems)
        missing_labels = sorted(img_stems - lbl_stems)

        present_seqs = set()
        for stem in lbl_stems:
            if "_" in stem:
                present_seqs.add(stem.split("_", 1)[0])

        missing_seqs = sorted(expected[split] - present_seqs)

        split_ok = len(lbl_stems) > 0 and not missing_images and not missing_labels and not missing_seqs
        if not split_ok:
            result["ok"] = False
            if len(lbl_stems) == 0:
                result["errors"].append(f"Split '{split}' has no labels.")
            if missing_images:
                result["errors"].append(
                    f"Split '{split}' has labels without images: {missing_images[:5]}"
                )
            if missing_labels:
                result["errors"].append(
                    f"Split '{split}' has images without labels: {missing_labels[:5]}"
                )
            if missing_seqs:
                result["errors"].append(
                    f"Split '{split}' is missing expected sequences: {missing_seqs}"
                )

        result["splits"][split] = {
            "images": len(img_stems),
            "labels": len(lbl_stems),
            "missing_images": len(missing_images),
            "missing_labels": len(missing_labels),
            "missing_sequences": missing_seqs,
        }

    return result

if __name__ == "__main__":
    # Example usage logic
    pass
