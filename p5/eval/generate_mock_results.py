import json
import random
import copy
import os

def generate_mock_preds(gt_json_path, output_path, confidence_range=(0.5, 0.99), noise_level=0.05):
    """
    Generates mock predictions from ground truth JSON.
    noise_level: amount of random offset added to bbox coordinates.
    """
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    preds = []
    for anno in gt_data['annotations']:
        # Copy bbox and add noise
        bbox = anno['bbox']
        noisy_bbox = [
            bbox[0] + random.uniform(-noise_level, noise_level) * bbox[2],
            bbox[1] + random.uniform(-noise_level, noise_level) * bbox[3],
            bbox[2] * random.uniform(1 - noise_level, 1 + noise_level),
            bbox[3] * random.uniform(1 - noise_level, 1 + noise_level)
        ]
        
        pred = {
            "image_id": anno['image_id'],
            "category_id": anno['category_id'],
            "bbox": noisy_bbox,
            "score": random.uniform(*confidence_range)
        }
        preds.append(pred)
    
    # Randomly drop some detections to simulate false negatives
    preds = random.sample(preds, int(len(preds) * 0.95))
    
    # S'assurer que le dossier de destination existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(preds, f)
    print(f"Mock predictions saved to {output_path}")

if __name__ == "__main__":
    # Example logic
    pass
