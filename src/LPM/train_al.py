import os
import sys

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
    default_data_collator,
)

import wandb
from models import LPM
from utils import MarginRankingLoss_learning_loss, progress_bar

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataloader import load

wandb.init(project="em_sentiment")

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

    # loss function
    loss_pred_module = LPM.loss_prediction_module(
        cfg.DATASETS.seq_len,
        cfg.MODEL.outputs_size,
        model.config.num_hidden_layers,
    )
    loss_pred_module = loss_pred_module.to(device)

    # optimizer
    optimizer_target = optim.Adam(
        model.parameters(),
        lr=cfg.TRAININGS.learning_rate,
        weight_decay=5e-4,
    )
    optimizer_loss = optim.Adam(
        loss_pred_module.parameters(),
        lr=cfg.TRAININGS.learning_rate,
        weight_decay=5e-4,
    )

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss_pred_criterion = MarginRankingLoss_learning_loss()

    # wandb
    wandb.watch(model, log_freq=100)

    # train
    print(f"model train start...", end=" ")

    def train(epoch):
        print("\nEpoch: %d" % epoch)
        model.train()
        loss_pred_module.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(trainloader):
            batch = {k: v.cuda() for k, v in batch.items()}

            optimizer_target.zero_grad()
            optimizer_loss.zero_grad()

            outputs = model(output_hidden_states=True, **batch)
            loss_pred = loss_pred_module(outputs.hidden_states)
            loss = criterion(outputs.logits, batch["labels"])
            loss = loss.view([-1, 1])

            loss_prediction_loss = loss_pred_criterion(loss_pred, loss)
            target_loss = loss.mean()

            if epoch < 120:
                loss = loss_prediction_loss + target_loss
                loss.backward()
                optimizer_target.step()
                optimizer_loss.step()
            else:
                loss = target_loss
                loss.backward()
                optimizer_target.step()

            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += batch["labels"].size(0)
            correct += predicted.eq(batch["labels"]).sum().item()

            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
            if epoch < 120:
                wandb.log(
                    {
                        "train/loss": train_loss / (batch_idx + 1),
                        "train/lpm_loss": loss_prediction_loss,
                        "train/target_loss": target_loss,
                        "train/acc": 100.0 * correct / total,
                    }
                )
            else:
                wandb.log(
                    {
                        "train/loss": train_loss / (batch_idx + 1),
                        "train/target_loss": target_loss,
                        "train/acc": 100.0 * correct / total,
                    }
                )

    # test
    def test(epoch):
        global best_acc
        model.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(output_hidden_states=True, **batch)
                loss = criterion(outputs.logits, batch["labels"])

                loss = loss.mean()
                test_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                total += batch["labels"].size(0)
                correct += predicted.eq(batch["labels"]).sum().item()

                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                wandb.log(
                    {
                        "eval/loss": test_loss / (batch_idx + 1),
                        "eval/acc": 100.0 * correct / total,
                    }
                )

        # Save checkpoint.
        acc = 100.0 * correct / total
        if acc > best_acc:
            print("Saving..")
            state = {
                "model": model.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, cfg.AL.checkpoint_dir)
            model.save_pretrained(os.path.join(cfg.AL.lpm_dir, "ckpt"))
            torch.save(loss_pred_module, os.path.join(cfg.AL.lpm_dir, "lpm.pt"))
            best_acc = acc

    start_epoch = cfg.AL.num_train_epochs
    for epoch in range(start_epoch + 1):
        train(epoch)
        test(epoch)
    print("model train done!")


if __name__ == "__main__":
    main()
