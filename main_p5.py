import os
from p5.data.download_mot17 import download_mot17
from p5.data.create_dummy_data import create_dummy_mot17
from p5.data.mot_to_yolo import convert_mot_to_yolo
from p5.data.mot_to_coco import mot_to_coco
from p5.data.split_data import split_sequences
from p5.eval.evaluate import evaluate_coco, compare_models
from p5.eval.generate_mock_results import generate_mock_preds

def run_p5_pipeline(mot_root='data/raw/MOT17'):
    print("Starting Person 5 Pipeline...")
    
    # 0. Download dataset if missing
    if not os.path.exists(mot_root):
        try:
            download_mot17('data/raw')
        except Exception as e:
            print(f"\n[AVERTISSEMENT] Échec du téléchargement : {e}")
            print("Utilisation de données factices (Dummy Data) pour continuer le test du pipeline.\n")
            create_dummy_mot17(mot_root)
    
    # 1. Dataset Formatting & Splitting
    if not os.path.exists(mot_root):
        print(f"Error: {mot_root} not found even after download attempt.")
        return

    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Split sequences
    train_seqs, val_seqs = split_sequences(os.path.join(mot_root, 'train'))
    print(f"Train sequences: {train_seqs}")
    print(f"Val sequences: {val_seqs}")
    
    # Convert to YOLO (for P2)
    convert_mot_to_yolo(mot_root, 'data/processed/MOT17_YOLO')
    
    # Convert to COCO (for P3 and Evaluation)
    mot_to_coco(mot_root, 'data/processed/val_gt.json', val_seqs)
    mot_to_coco(mot_root, 'data/processed/train_gt.json', train_seqs)

    # 2. Evaluation (using mock results for now)
    print("\nRunning Evaluation...")
    gt_path = 'data/processed/val_gt.json'
    if os.path.exists(gt_path):
        # Simulate P2 and P3 results
        generate_mock_preds(gt_path, 'results/p2/preds.json', confidence_range=(0.7, 0.95), noise_level=0.03)
        generate_mock_preds(gt_path, 'results/p3/preds.json', confidence_range=(0.6, 0.90), noise_level=0.05)
        
        # Evaluate
        stats_p2 = evaluate_coco(gt_path, 'results/p2/preds.json')
        stats_p3 = evaluate_coco(gt_path, 'results/p3/preds.json')
        
        # Compare
        compare_models({'YOLOv11': stats_p2, 'DETR': stats_p3})
    else:
        print("Val GT not found. Create it by running the script with real data.")

if __name__ == "__main__":
    run_p5_pipeline()
