import albumentations as A
import cv2
import os

def get_augmentation_pipeline():
    """
    Defines a simple augmentation pipeline for object detection.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_image(image_path, label_path, output_img_path, output_lbl_path):
    """
    Applies augmentation to a single image and its labels.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_labels.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])
            
    transform = get_augmentation_pipeline()
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    cv2.imwrite(output_img_path, transformed['image'])
    with open(output_lbl_path, 'w') as f:
        for i in range(len(transformed['bboxes'])):
            bbox = transformed['bboxes'][i]
            cls = transformed['class_labels'][i]
            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in bbox])}\n")

if __name__ == "__main__":
    pass
