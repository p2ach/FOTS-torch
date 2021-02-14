import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import numpy as np

from eval_tools.icdar2015 import eval as icdar_eval

from trainer import Train
from data_helpers.datasets import ICDARDataset
from data_helpers.data_utils import icdar_collate
from components.loss import FOTSLoss
from model import FOTSModel
from trainer import Train


def fots_metric(pred, gt):
    config = icdar_eval.default_evaluation_params()
    output = icdar_eval.eval(pred, gt, config)
    return output['method']['precision'], output['method']['recall'], output['method']['hmean']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main entry point of train module."""
    # Initialize the dataset
    # Full dataset
    icdar_dataset = ICDARDataset('/content/ch4_training_images', '/content/ch4_training_localization_transcription_gt')

    # Train test split
    val_size = 0.12
    val_len = int(val_size * len(icdar_dataset))
    train_len = len(icdar_dataset) - val_len
    icdar_train_dataset, icdar_val_dataset = torch.utils.data.random_split(
        icdar_dataset, [train_len, val_len]
    )

    icdar_train_data_loader = DataLoader(
        icdar_train_dataset,
        num_workers=4,
        batch_size=16,
        shuffle=False,
        collate_fn=icdar_collate
    )

    icdar_val_data_loader = DataLoader(
        icdar_val_dataset,
        num_workers=4,
        batch_size=16,
        shuffle=False,
        collate_fn=icdar_collate
    )

    # Initialize the model
    model = FOTSModel()

    # Count trainable parameters
    print(f'The model has {count_parameters(model):,} trainable parameters.')

    loss = FOTSLoss()

    optimizer = optim.Adam(model.parameters())
    
    trainer = Train(
        model,
        icdar_train_data_loader,
        icdar_val_data_loader,
        loss,
        fots_metric,
        optimizer,
        12
    )

    trainer.train()


if __name__ == '__main__':
    main()
