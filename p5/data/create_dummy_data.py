import os
import cv2
import numpy as np

def create_dummy_mot17(base_path='data/raw/MOT17'):
    """
    Crée une structure MOT17 minimale pour tester le pipeline.
    """
    print("Création d'un dataset MOT17 factice pour test...")
    sequences = ['MOT17-02-DPM', 'MOT17-04-DPM']
    
    for seq in sequences:
        # Chemins
        seq_path = os.path.join(base_path, 'train', seq)
        img_dir = os.path.join(seq_path, 'img1')
        gt_dir = os.path.join(seq_path, 'gt')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        
        # Créer 5 images vides (noires)
        for i in range(1, 6):
            img_name = f"{i:06d}.jpg"
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(img, f"Sequence {seq} Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(img_dir, img_name), img)
            
        # Créer un fichier gt.txt (format: frame, id, x, y, w, h, conf, class, visibility)
        gt_content = [
            f"1,1,100,100,50,100,1,1,1.0",
            f"2,1,110,100,50,100,1,1,1.0",
            f"3,1,120,100,50,100,1,1,1.0",
            f"4,1,130,100,50,100,1,1,1.0",
            f"5,1,140,100,50,100,1,1,1.0"
        ]
        with open(os.path.join(gt_dir, 'gt.txt'), 'w') as f:
            f.write('\n'.join(gt_content))

    # Créer un fichier seqinfo.ini minimal (optionnel mais propre)
    print("Structure factice créée avec succès.")

if __name__ == "__main__":
    create_dummy_mot17()
