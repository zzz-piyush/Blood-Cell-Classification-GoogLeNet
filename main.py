"""
Main Entry Point -- GoogLeNet Blood Cell Classification
========================================================
Usage:
    python main.py                    # Full pipeline (train + evaluate)
    python main.py --mode train       # Train only
    python main.py --mode evaluate    # Evaluate only (requires checkpoint)
"""

import os
import argparse
import yaml
import torch

from src.model import GoogLeNet, get_model_summary
from src.dataset import download_dataset, create_dataloaders
from src.train import run_training
from src.evaluate import run_evaluation
from src.utils import set_seed, load_checkpoint, plot_training_curves


def load_config(path="configs/config.yaml"):
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GoogLeNet Blood Cell Classifier")
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'evaluate', 'full'],
                        help='Run mode: train, evaluate, or full pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directories
    for key in ['dir', 'checkpoint_dir', 'plot_dir', 'metrics_dir',
                'log_dir', 'gradcam_dir']:
        os.makedirs(config['output'][key], exist_ok=True)

    # Download dataset
    data_dir = download_dataset(
        config['dataset']['name'],
        config['dataset']['data_dir']
    )

    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir,
        image_size=config['dataset']['image_size'],
        batch_size=config['training']['batch_size'],
        val_split=config['dataset']['val_split'],
        num_workers=config['dataset']['num_workers'],
        seed=config['seed']
    )

    # Build model
    model = GoogLeNet(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        aux_dropout=config['model']['aux_dropout'],
        use_batchnorm=config['model']['use_batchnorm']
    ).to(device)

    get_model_summary(model)

    # ----- TRAIN -----
    if args.mode in ('train', 'full'):
        history = run_training(
            model, train_loader, val_loader, config, device,
            config['output']['checkpoint_dir'],
            config['output']['log_dir']
        )
        plot_training_curves(history, config['output']['plot_dir'])

    # ----- EVALUATE -----
    if args.mode in ('evaluate', 'full'):
        checkpoint_path = os.path.join(config['output']['checkpoint_dir'],
                                        'best_model.pth')
        if os.path.exists(checkpoint_path):
            print(f"\n  Loading best model from: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path, device)
            model.to(device)
        else:
            print("  Warning: No checkpoint found, evaluating current model")

        metrics = run_evaluation(model, test_loader, class_names, device, config)

    print("\n  All done!")


if __name__ == "__main__":
    main()
