import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import yaml
import glob
import csv
import time

# Import the model definition from your training script
from train import DinoV3Segmenter

# --- 1. IoU Calculation Function ---
def calculate_mask_iou(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.
    Masks are expected to be boolean numpy arrays.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# --- 2. False Positive Calculation ---
def calculate_false_positives(model, image_dir, conf_threshold, output_dir, image_transform, device):
    """
    Calculates the number of images with false positive detections and saves visualizations.
    """
    if not os.path.isdir(image_dir): return "N/A"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files: return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    false_positive_count = 0
    model.eval()
    with torch.no_grad():
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            input_tensor = image_transform(image).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            pred_mask = (torch.sigmoid(output) > conf_threshold).squeeze().cpu().numpy()

            # If there is any detection, count it as a false positive
            if np.sum(pred_mask) > 0:
                false_positive_count += 1
                
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                pred_mask_colored = np.zeros_like(image_cv)
                pred_mask_resized_for_viz = cv2.resize(pred_mask.astype(np.uint8), (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred_mask_colored[pred_mask_resized_for_viz == 1] = [0, 0, 255] # Red for FP

                overlaid_image = cv2.addWeighted(image_cv, 0.7, pred_mask_colored, 0.3, 0)
                save_path = os.path.join(output_dir, image_file)
                cv2.imwrite(save_path, overlaid_image)
                
    return false_positive_count

# --- 3. Visualization Function ---
def visualize_output(image, pred_mask, gt_mask):
    """Draws predicted and ground truth masks on an image."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Green for prediction
    pred_mask_colored = np.zeros_like(image_cv)
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask_colored[pred_mask_resized == 1] = [0, 255, 0]
    
    # Blue for ground truth
    gt_mask_colored = np.zeros_like(image_cv)
    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask_colored[gt_mask_resized == 1] = [255, 0, 0]

    # Blend all together
    overlaid_image = cv2.addWeighted(image_cv, 1.0, pred_mask_colored, 0.5, 0)
    overlaid_image = cv2.addWeighted(overlaid_image, 1.0, gt_mask_colored, 0.5, 0)
    
    return overlaid_image

# --- 4. Main Benchmarking Function ---
def main():
    # --- Config ---
    RESULTS_DIR = "./results"
    VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
    FP_VIZ_DIR = os.path.join(RESULTS_DIR, "false_positives")
    MODEL_ROOT_DIR = "./output"
    CONFIG_DIR = "./configs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(FP_VIZ_DIR, exist_ok=True)

    config_files = glob.glob(os.path.join(CONFIG_DIR, '**/*.yaml'), recursive=True)
    if not config_files:
        print(f"No .yaml config files found in {CONFIG_DIR}. Nothing to benchmark.")
        return

    csv_output_path = os.path.join(RESULTS_DIR, "dinov3_benchmark_results.csv")
    csv_headers = ["Model", "Run Name", "Epochs", "lr0", "batch", "conf", "mask_iou", "avg_inference_time", "detection_failed", "false_positives"]
    all_csv_rows = []

    for config_path in sorted(config_files):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        run_name = config['output']['run_name']
        model_dir = os.path.join(MODEL_ROOT_DIR, run_name)
        model_path = os.path.join(model_dir, 'best_model.pth')

        if not os.path.exists(model_path):
            print(f"\nSkipping {run_name}: Trained model not found at {model_path}")
            continue

        print(f"\n--- Benchmarking run: {run_name} ---")

        # --- Load Model ---
        model = DinoV3Segmenter(
            model_name=config['model']['name'], 
            weights_path=config['model']['weights'],
            num_classes=config['model']['num_classes'],
            head_channels=config['model'].get('head_channels')
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # --- Data and Transforms ---
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_image_dir = config['dataset']['test_images']
        test_mask_dir = config['dataset']['test_masks']
        reject_image_dir = config['dataset']['reject_images']
        test_images = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # --- Evaluation ---
        conf_threshold = config['benchmarking']['threshold']
        viz_output_dir = os.path.join(VISUALIZATION_DIR, f"{run_name}_{conf_threshold}")
        os.makedirs(viz_output_dir, exist_ok=True)

        total_iou = 0.0
        inference_times = []
        detections_failed = 0

        with torch.no_grad():
            for img_name in tqdm(test_images, desc=f"Evaluating at conf={conf_threshold}"):
                img_path = os.path.join(test_image_dir, img_name)
                mask_path = os.path.join(test_mask_dir, img_name)

                if not os.path.exists(mask_path): continue

                image = Image.open(img_path).convert("RGB")
                gt_mask_pil = Image.open(mask_path).convert("L")
                gt_mask_np = np.array(gt_mask_pil) > 128

                input_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

                start_time = time.time()
                output = model(input_tensor)
                inference_times.append(time.time() - start_time)

                # Resize predicted mask to match ground truth mask size
                pred_mask_tensor = (torch.sigmoid(output) > conf_threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred_mask_pil = Image.fromarray(pred_mask_tensor * 255)
                pred_mask_resized_pil = pred_mask_pil.resize(gt_mask_pil.size, Image.NEAREST)
                pred_mask_np = np.array(pred_mask_resized_pil) > 128

                iou = calculate_mask_iou(pred_mask_np, gt_mask_np)
                total_iou += iou
                if np.sum(pred_mask_np) == 0 and np.sum(gt_mask_np) > 0:
                    detections_failed += 1

                # Save visualization
                viz_image = visualize_output(image, pred_mask_np, gt_mask_np)
                cv2.imwrite(os.path.join(viz_output_dir, img_name), viz_image)

        avg_iou = total_iou / len(test_images) if test_images else 0
        avg_time = np.mean(inference_times) if inference_times else 0

        # --- False Positives ---
        fp_output_dir = os.path.join(FP_VIZ_DIR, f"{run_name}_{conf_threshold}")
        false_positives = calculate_false_positives(model, reject_image_dir, conf_threshold, fp_output_dir, image_transform, DEVICE)

        # --- Record Results ---
        row = {
            "Model": config['model']['name'],
            "Run Name": run_name,
            "Epochs": config['training']['epochs'],
            "lr0": config['training']['optimizer']['lr0'],
            "batch": config['training']['batch_size'],
            "conf": conf_threshold,
            "mask_iou": f"{avg_iou:.4f}",
            "avg_inference_time": f"{avg_time:.4f}",
            "detection_failed": detections_failed,
            "false_positives": false_positives
        }
        all_csv_rows.append(row)
        print(f"Finished benchmarking {run_name}. Results added to list.")

    if all_csv_rows:
        try:
            with open(csv_output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(all_csv_rows)
            print(f"\nAll benchmark results saved to: {csv_output_path}")
        except IOError as e:
            print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    main()
