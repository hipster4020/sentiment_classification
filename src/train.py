import json
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, logging
from torch import nn
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from dataloader import load

logging.set_verbosity(logging.ERROR)


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer_dir)

    # dataloder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    print(train_dataset[0])
    print(eval_dataset)

    # args = TrainingArguments(
    #     num_train_epochs=trainings["epochs"],
    #     per_device_train_batch_size=trainings["train_batch_size"],
    #     per_device_eval_batch_size=trainings["eval_batch_size"],
    #     learning_rate=trainings["learning_rate"],
    #     adam_beta1=trainings["adam_beta1"],
    #     adam_beta2=trainings["adam_beta2"],
    #     adam_epsilon=trainings["adam_epsilon"],
    #     warmup_steps=trainings["warmup_steps"],
    #     logging_dir=trainings["logging_dir"],
    #     logging_steps=trainings["logging_steps"],
    #     seed=trainings["seed"],
    #     output_dir=etc["output_dir"],
    # )

    # pretrained_model_config = BertConfig.from_pretrained(
    #     models["pretrained_model_name"],
    #     num_labels=models["num_labels"],
    # )

    # model = BertForSequenceClassification.from_pretrained(
    #     models["pretrained_model_name"],
    #     config=pretrained_model_config,
    # )

    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset,
    #     nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0])),
    #     eval_dataset=eval_dataset,
    # )


if __name__ == "__main__":
    main()
