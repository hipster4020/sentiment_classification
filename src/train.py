import hydra
import numpy as np
import wandb
from datasets import load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from dataloader import load

wandb.init(project="em_sentiment")


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)

    # data loder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    args = TrainingArguments(
        **cfg.TRAININGS,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL.pretrained_model_name,
        num_labels=cfg.MODEL.num_labels,
    )

    # metrics
    def compute_metrics(eval_preds):
        metric = load_metric(cfg.METRICS.metric_name)
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return metric.compute(
            predictions=predictions,
            references=labels,
            average=cfg.METRICS.average,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(cfg.PATH.save_dir)


if __name__ == "__main__":
    main()
