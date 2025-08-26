import yaml
import shutil
import os
import torch
import random
import numpy as np
import glob # To find all config files
from ultralytics import YOLO

def main():
    """
    Finds and runs all YOLO training configurations located in the 'configs' directory.
    """
    config_dir = 'configs'
    config_files = glob.glob(os.path.join(config_dir, '**/*.yaml'), recursive=True)

    if not config_files:
        print(f"Error: No .yaml configuration files found in the '{config_dir}/' directory.")
        return
        
    print(f"Found {len(config_files)} training configurations to run.")

    for i, config_path in enumerate(sorted(config_files)):
        print("\n" + "="*80)
        print(f"STARTING TRAINING RUN {i+1}/{len(config_files)}: {config_path}")
        print("="*80 + "\n")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_path}. Skipping.")
            continue
        
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Initializing model: {config['model']}")
        model = YOLO(config['model'])

        train_args = {
            'data': config['data'],
            'epochs': config.get('epochs', 100),
            'batch': config.get('batch', 16),
            'lr0': config.get('lr0', 0.01),
            'freeze': config.get('freeze', None),
            'project': config.get('project', 'output'),
            'name': config.get('name', 'exp'),
            'exist_ok': True,
        }

        if config.get('augment', False):
            print("Applying augmentations...")
            train_args.update({
                'degrees': config.get('degrees', 0),
                'translate': config.get('translate', 0),
                'scale': config.get('scale', 0),
                'shear': config.get('shear', 0),
                'perspective': config.get('perspective', 0),
            })

        print(f"Starting training for run: {train_args['name']}")
        results = model.train(**train_args)

        print(f"\nTraining complete. Results saved to: {results.save_dir}")

    print("\n\nAll training runs have finished successfully!")

if __name__ == '__main__':
    main()