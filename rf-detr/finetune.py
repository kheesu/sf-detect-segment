import yaml
import os
import shutil
import torch
import random
import numpy as np
import glob # Added to find files

# Assuming your rfdetr module is in the src directory.
# Add the project root to the Python path if needed:
# import sys
# sys.path.append(os.getcwd())
from rfdetr import RFDETRMedium, RFDETRSmall

def main():
    config_dir = 'configs'
    config_files = glob.glob(os.path.join(config_dir, '**/*.yaml'), recursive=True)
    config_files.extend(glob.glob(os.path.join(config_dir, '**/*.yml'), recursive=True))

    if not config_files:
        print(f"Error: No .yaml or .yml configuration files found in the '{config_dir}/' directory.")
        return
        
    for i, config_path in enumerate(sorted(config_files)): # Sorting for consistent order
        print("\n" + "="*80)
        print(f"STARTING TRAINING {i+1}/{len(config_files)}: {config_path}")
        print("="*80 + "\n")

        # --- Load Configuration ---
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}. Skipping run.")
            continue
            
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model_config = config['model']
        train_config = config['training']
        data_config = config['dataset']
        output_config = config['output']

        model_size = model_config['size']
        num_classes = model_config['num_classes']
        epochs = train_config['epochs']
        batch_size = train_config['batch_size']
        grad_accum_steps = train_config['gradient_accumulation_steps']
        lr = train_config['optimizer']['lr0']
        
        dataset_dir = os.path.dirname(data_config['train_path'])
        output_dir = os.path.join(output_config['save_dir'], f'{model_size}_{epochs}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initializing RF-DETR '{model_size}' model...")
        if model_size == 'medium':
            model = RFDETRMedium(num_classes=num_classes)
        else:
            model = RFDETRSmall(num_classes=num_classes)

        print(f"Starting training for {epochs} epochs...")
        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=output_dir,
        )

        shutil.copy(config_path, os.path.join(output_dir, 'args.yaml'))
        print(f"\nTraining run for {config_path} complete. Model saved to: {output_dir}")

    print("\n\nAll training runs have finished")


if __name__ == '__main__':
    main()