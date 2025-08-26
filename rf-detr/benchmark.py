import os
import time
import torch
import cv2
import numpy as np
import random
import csv
import json
import glob
import yaml

from rfdetr import RFDETRMedium, RFDETRSmall
from segment_anything import sam_model_registry, SamPredictor

def calculate_mask_iou(mask1, mask2):
    """Calculates the Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0
    return np.sum(intersection) / np.sum(union)

def calculate_bbox_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = xB - xA
    inter_height = yB - yA
    if inter_width <= 0 or inter_height <= 0:
        intersection_area = 0
    else:
        intersection_area = inter_width * inter_height
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_false_positives(model, image_dir, conf_threshold, output_dir):
    """
    Calculates the number of images with false positive detections and saves visualizations.
    """
    if not os.path.isdir(image_dir): return "N/A"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files: return 0
    
    # Create the output directory for this run's false positives
    os.makedirs(output_dir, exist_ok=True)
    
    false_positive_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        detections = model.predict(image_path)
        
        if detections.xyxy is not None:
            confident_indices = [i for i, score in enumerate(detections.confidence) if score >= conf_threshold]
            if confident_indices:
                false_positive_count += 1
                
                # Save visualization of the false positive
                image_bgr = cv2.imread(image_path)
                # Loop through all confident detections for this image
                for i in confident_indices:
                    box = detections.xyxy[i].astype(int)
                    # Draw a red box for the false positive detection
                    cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                
                # Save the image with overlaid box(es)
                save_path = os.path.join(output_dir, image_file)
                cv2.imwrite(save_path, image_bgr)
                
    return false_positive_count

def visualize_output(image, mask, pred_box, gt_box):
    """Draws mask and bounding boxes on an image."""
    color = np.array([0, 255, 0], dtype=np.uint8) # Green
    masked_image = np.where(mask[..., None], color, image)
    overlaid_image = cv2.addWeighted(image, 0.6, masked_image, 0.4, 0)
    cv2.rectangle(overlaid_image, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 2) # Red
    cv2.rectangle(overlaid_image, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0, 255, 0), 2) # Green
    return overlaid_image

def main():
    # --- 1. Configuration ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    RESULTS_DIR = "./results"
    VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
    FP_VIZ_DIR = os.path.join(RESULTS_DIR, "false_positives") # Directory for FP images
    IMAGE_DIR = os.path.abspath("../data/coco_dataset_bbox/test")
    GT_MASK_DIR = os.path.abspath("../data/gt/test") # <-- Use masks from this directory
    REJECT_IMAGE_DIR = os.path.abspath("../data/reject")
    CONFIG_DIR = "./configs"
    MODEL_ROOT_DIR = "./output"
    ANNOTATION_FILE = os.path.join(IMAGE_DIR, "_annotations.coco.json")
    SAM_CHECKPOINT = "../sam_vit_h_4b8939.pth"
    SAM_MODEL_NAME = "sam_vit_h_4b8939"
    SAM_MODEL_TYPE = "vit_h"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLDS = [0.25, 0.01, 0.001]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(FP_VIZ_DIR, exist_ok=True) # Create the main FP directory

    # --- 2. Load Models and Data ---
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"SAM checkpoint not found at {SAM_CHECKPOINT}. Cannot proceed.")
        return
    print("Loading SAM...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print("SAM loaded.")

    with open(ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
    images = {img['id']: img for img in coco_data['images']}
    annotations = {img_id: [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id] for img_id in images}
    print(f"Found {len(images)} images and {len(coco_data['annotations'])} annotations.")

    config_files = glob.glob(os.path.join(CONFIG_DIR, '**/*.yaml'), recursive=True)
    if not config_files:
        print(f"No .yaml config files found in {CONFIG_DIR}. Nothing to benchmark.")
        return
    
    csv_output_path = os.path.join(RESULTS_DIR, "rfdetr_benchmark_results.csv")
    csv_headers = ["Model", "Dataset", "Epochs", "lr0", "batch", "frozen layers", "conf", "segmentation model", "mask IoU (bbox)", "mask IoU (point)", "bbox IoU", "mAP@0.5", "mAP@0.75", "mAP@[0.5:0.05:0.95]", "avg inference time (with sam)", "detection failed", "false positives (out of 100)"]
    all_csv_rows = []
    
    for config_path in sorted(config_files):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_size = config['model']['size']
        model_epochs = config['training']['epochs']
        run_name = f"{model_size}_{model_epochs}"
        model_dir = os.path.join(MODEL_ROOT_DIR, run_name)
        model_path = os.path.join(model_dir, 'checkpoint_best_total.pth')

        if not os.path.exists(model_path):
            print(f"\nSkipping {config_path}: Trained model not found at {model_path}")
            continue
        
        print(f"\n--- Benchmarking run: {run_name} ---")
        
        for conf in THRESHOLDS:
            viz_output_dir = os.path.join(VISUALIZATION_DIR, f"{run_name}_{conf}")
            os.makedirs(viz_output_dir, exist_ok=True)
        print(f"Visualizations will be saved to subdirectories in: {VISUALIZATION_DIR}")

        try:
            rfdetr_model = RFDETRSmall(pretrain_weights=model_path) if 'small' in model_size else RFDETRMedium(pretrain_weights=model_path)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue

        results_by_threshold = {t: {'mask_ious_bbox_prompt': [], 'mask_ious_point_prompt': [], 'bbox_ious': []} for t in THRESHOLDS}
        rfdetr_times, sam_times = [], []

        for image_id, image_info in list(images.items()):
            if not annotations.get(image_id): continue
            
            image_file, img_h, img_w = image_info['file_name'], image_info['height'], image_info['width']
            image_path = os.path.join(IMAGE_DIR, image_file)
            if not os.path.exists(image_path): continue
            
            # --- MODIFIED SECTION: Load GT mask from file ---
            mask_path = os.path.join(GT_MASK_DIR, image_file)
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found for {image_file}, skipping.")
                continue
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Warning: Could not read mask file for {image_file}, skipping.")
                continue
            gt_mask = gt_mask > 128
            # --- END MODIFIED SECTION ---
            
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            ann = annotations[image_id][0]
            gt_bbox = ann['bbox']
            gt_bbox_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

            start_time = time.time()
            detections = rfdetr_model.predict(image_path)
            rfdetr_times.append(time.time() - start_time)
            predictor.set_image(image_rgb)

            for conf in THRESHOLDS:
                confident_indices = [i for i, c in enumerate(detections.confidence) if c >= conf]
                if confident_indices:
                    best_index = confident_indices[np.argmax(detections.confidence[confident_indices])]
                    pred_bbox = detections.xyxy[best_index]
                    
                    viz_output_dir = os.path.join(VISUALIZATION_DIR, f"{run_name}_{conf}")
                    masks_viz, _, _ = predictor.predict(box=np.array(pred_bbox), multimask_output=False)
                    overlaid_image = visualize_output(image_bgr, masks_viz[0], pred_bbox, gt_bbox_xyxy)
                    cv2.imwrite(os.path.join(viz_output_dir, image_file), overlaid_image)
                    
                    results_by_threshold[conf]['bbox_ious'].append(calculate_bbox_iou(pred_bbox, gt_bbox_xyxy))
                    
                    start_sam = time.time()
                    masks_bbox, _, _ = predictor.predict(box=np.array(pred_bbox), multimask_output=False)
                    sam_times.append(time.time() - start_sam)
                    results_by_threshold[conf]['mask_ious_bbox_prompt'].append(calculate_mask_iou(masks_bbox[0], gt_mask))

                    center_point = np.array([[(pred_bbox[0] + pred_bbox[2])/2, (pred_bbox[1] + pred_bbox[3])/2]])
                    start_sam = time.time()
                    masks_point, _, _ = predictor.predict(point_coords=center_point, point_labels=np.array([1]), multimask_output=False)
                    sam_times.append(time.time() - start_sam)
                    results_by_threshold[conf]['mask_ious_point_prompt'].append(calculate_mask_iou(masks_point[0], gt_mask))
                else:
                    results_by_threshold[conf]['bbox_ious'].append(0.0)
                    results_by_threshold[conf]['mask_ious_bbox_prompt'].append(0.0)
                    results_by_threshold[conf]['mask_ious_point_prompt'].append(0.0)

        avg_inference_time = (np.mean(rfdetr_times) if rfdetr_times else 0) + (np.mean(sam_times) if sam_times else 0)
        for conf in THRESHOLDS:
            data = results_by_threshold[conf]
            num_preds = len(data['bbox_ious'])
            if num_preds == 0: continue
            
            aps = [sum(1 for s in data['bbox_ious'] if s >= t) / num_preds for t in np.arange(0.5, 1.0, 0.05)]
            
            fp_output_dir = os.path.join(FP_VIZ_DIR, f"{run_name}_{conf}")
            
            row = {
                "Model": model_size,
                "Dataset": os.path.basename(os.path.dirname(ANNOTATION_FILE)),
                "Epochs": model_epochs,
                "lr0": config['training']['optimizer']['lr0'],
                "batch": config['training']['batch_size'],
                "frozen layers": config['training'].get('freeze', 'N/A'),
                "conf": conf,
                "segmentation model": SAM_MODEL_NAME,
                "mask IoU (bbox)": f"{np.mean(data['mask_ious_bbox_prompt']):.4f}",
                "mask IoU (point)": f"{np.mean(data['mask_ious_point_prompt']):.4f}",
                "bbox IoU": f"{np.mean(data['bbox_ious']):.4f}",
                "mAP@0.5": f"{aps[0]:.4f}",
                "mAP@0.75": f"{aps[5]:.4f}",
                "mAP@[0.5:0.05:0.95]": f"{np.mean(aps):.4f}",
                "avg inference time (with sam)": f"{avg_inference_time:.4f}",
                "detection failed": sum(1 for iou in data['bbox_ious'] if iou == 0.0),
                "false positives (out of 100)": calculate_false_positives(rfdetr_model, REJECT_IMAGE_DIR, conf, fp_output_dir)
            }
            all_csv_rows.append(row)
        print(f"Finished benchmarking {run_name}. Results added to main list.")

    if all_csv_rows:
        try:
            with open(csv_output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers, delimiter=',')
                writer.writeheader()
                writer.writerows(all_csv_rows)
            print(f"\nAll benchmark results successfully saved to a single file: {csv_output_path}")
        except IOError as e:
            print(f"Error writing final CSV file: {e}")

if __name__ == "__main__":
    main()
