import argparse
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_metric
from pshmodule.utils import filemanager as fm
from tensorflow.keras.models import load_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ElectraForSequenceClassification,
    default_data_collator,
)

from dataloader import load
from models import LPM
from utils import MarginRankingLoss_learning_loss, progress_bar

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0
start_epoch = 0


@hydra.main(config_name="config.yml")
def main(cfg):
    # electra
    # tokenizer & model
    print(f"load tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)
    print("tokenizer loading done!")

    # data loader
    print(f"data loader...", end=" ")
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    print(f"len(train_dataset) : {len(train_dataset)}")

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAININGS.per_device_train_batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
        num_workers=2,
    )
    testloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.TRAININGS.per_device_eval_batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
        num_workers=2,
    )
    print("data loader done!")

    # model
    print(f"load model...", end=" ")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL.pretrained_model_name,
        num_labels=cfg.MODEL.num_labels,
    )
    model.cuda().eval()
    print("load model done!")

    print(model.electra.encoder.layer)


if __name__ == "__main__":
    main()
