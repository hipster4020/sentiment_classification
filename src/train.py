import json
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, logging
from torch import nn
from transformers import (AutoTokenizer, BertConfig,
                          BertForSequenceClassification, Trainer,
                          TrainingArguments, default_data_collator)

from dataloader import load

logging.set_verbosity(logging.ERROR)


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer_dir)

    # data loder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    
    args = TrainingArguments(
        num_train_epochs=cfg.TRAININGS.epochs,
        per_device_train_batch_size=cfg.TRAININGS.train_batch_size,
        per_device_eval_batch_size=cfg.TRAININGS.eval_batch_size,
        learning_rate=cfg.TRAININGS.learning_rate,
        adam_beta1=cfg.TRAININGS.adam_beta1,
        adam_beta2=cfg.TRAININGS.adam_beta2,
        adam_epsilon=cfg.TRAININGS.adam_epsilon,
        warmup_steps=cfg.TRAININGS.warmup_steps,
        logging_dir=cfg.TRAININGS.logging_dir,
        logging_steps=cfg.TRAININGS.logging_steps,
        seed=cfg.TRAININGS.seed,
        output_dir=cfg.PATH.output_dir,
    )

    pretrained_model_config = BertConfig.from_pretrained(
        cfg.MODEL.pretrained_model_name,
        num_labels=cfg.MODEL.num_labels,
    )

    model = BertForSequenceClassification.from_pretrained(
        cfg.MODEL.pretrained_model_name,
        config=pretrained_model_config,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
