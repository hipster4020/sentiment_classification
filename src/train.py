import json
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, logging
from torch import nn
from transformers import AutoTokenizer

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


if __name__ == "__main__":
    main()
