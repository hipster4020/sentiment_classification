import os
import sys

import hydra
import numpy as np
import torch
from pshmodule.utils import filemanager as fm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

from models import LPM

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataloader import load

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_name="config.yml")
def main(cfg):
    # electra
    # tokenizer & model
    print(f"load tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)
    print("tokenizer loading done!")

    # model
    print(f"load model...", end=" ")
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(cfg.AL.lpm_dir, "ckpt"),
        num_labels=cfg.MODEL.num_labels,
    )
    model.cuda().eval()
    print("load model done!")

    # lpm
    loss_pred_module = torch.load(os.path.join(cfg.AL.lpm_dir, "lpm.pt"))

    # data load
    df = fm.load(cfg.AL.untrain_data_path)
    df.dropna(axis=0, inplace=True)
    print(df.head())
    print(df.shape)

    # predict
    loss_list = []
    pred_list = []
    output_logits = []
    for sentence in df.content:
        data = tokenizer(
            sentence,
            max_length=cfg.DATASETS.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            data = {k: v.cuda() for k, v in data.items()}
            outputs = model(output_hidden_states=True, **data)

            # LPM
            loss_pred = loss_pred_module(outputs.hidden_states)
            loss_pred = round(loss_pred.tolist()[0][0], 4)
            loss_list.append(loss_pred)

            # predict
            predict = np.argmax(outputs.logits[0].cpu().numpy())

            sentiment = ""
            if predict == 0:
                sentiment = "중립"
            elif predict == 1:
                sentiment = "긍정"
            else:
                sentiment = "부정"
            pred_list.append(sentiment)
            output_logits.append(outputs.logits[0].cpu().numpy())

    df["lpm_loss"] = loss_list
    df["predict"] = pred_list
    df["logits"] = output_logits

    df = df[df.content.apply(lambda x: len(x) >= 50)]
    df.sort_values(by=["lpm_loss"], axis=0, ascending=False, inplace=True)
    print(f"df length : {len(df)}")

    # data save
    fm.save(cfg.AL.data_dir, df)


if __name__ == "__main__":
    main()
