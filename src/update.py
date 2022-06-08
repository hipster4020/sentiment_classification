import logging
import time
from logging import handlers

import hydra
import numpy as np
import pandas as pd
import swifter
import torch
from pshmodule.db import alchemy
from pshmodule.processing import processing as p
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# log setting
carLogFormatter = logging.Formatter("%(asctime)s,%(message)s")

carLogHandler = handlers.TimedRotatingFileHandler(
    filename="../../log/sentiment.log",
    when="midnight",
    interval=1,
    encoding="utf-8",
)

carLogHandler.setFormatter(carLogFormatter)
carLogHandler.suffix = "%Y%m%d"

scarp_logger = logging.getLogger()
scarp_logger.setLevel(logging.INFO)
scarp_logger.addHandler(carLogHandler)


# data load
def data_load(**kwargs):
    try:
        logging.info("dataload start")

        # start time
        start = time.time()

        df = alchemy.DataSource(
            {
                "id": kwargs.get("user"),
                "pwd": kwargs.get("passwd"),
                "ip": kwargs.get("host"),
                "port": kwargs.get("port"),
            },
            kwargs.get("db"),
        ).select_query_to_df(
            "select id, title, content from portal_news where create_date <= '2021-05-31'"
        )
        # end time
        logging.info("time :" + str(time.time() - start))

        # processing
        df["content"] = df.title + " " + df.content
        df["content"] = df.content.swifter.apply(p.news_preprocessing)
        df = df[["id", "content"]]

        logging.info(f"proccessing data : {df.head()}")
        logging.info(f"data shape : {df.shape}")

        logging.info("dataload end")

        return df

    except Exception as e:
        logging.info(e)


# excutemany update
def update(uquery, param, **kwargs):
    result = alchemy.DataSource(
        {
            "id": kwargs.get("user"),
            "pwd": kwargs.get("passwd"),
            "ip": kwargs.get("host"),
            "port": kwargs.get("port"),
        },
        kwargs.get("db"),
    ).executemany_query(uquery, param)
    logging.info(f"update result : {result}")
    logging.info("update end")


# batch 단위 나누기
def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


@hydra.main(config_name="config.yml")
def main(cfg):
    try:
        # data load
        df = data_load(**cfg.DATABASE)

        # electra
        # tokenizer & model
        logging.info(f"load tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)
        logging.info("tokenizer loading done!")

        logging.info(f"load model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.PATH.save_dir,
            num_labels=cfg.MODEL.num_labels,
        )
        model.eval().cuda()
        logging.info("model loading done!")

        # predict
        electra_pred = []
        for sentence in tqdm(batch(df.content, cfg.PREDICT.batch_size)):
            data = tokenizer(
                sentence.tolist(),
                max_length=cfg.DATASETS.seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                data = {k: v.cuda() for k, v in data.items()}
                outputs = model(**data)

                # encoding to sentiment wording
                for logit in outputs.logits:
                    predict = np.argmax(logit.cpu().numpy())
                    sentiment = ""
                    if predict == 0:
                        sentiment = "중립"
                    elif predict == 1:
                        sentiment = "긍정"
                    else:
                        sentiment = "부정"
                    electra_pred.append(sentiment)

        df["sentiment"] = electra_pred

        logging.info(f"after predict : {df.head()}")
        logging.info(f"predict shape : {df.shape}")

        # dataframe to update query
        query = "update portal_news set sentiment=%s where id=%s;"
        param = []
        for i in range(len(df)):
            temp = (df["sentiment"][i], df["id"][i])
            param.append(temp)

        update(query, param, **cfg.DATABASE)

    except Exception as e:
        logging.info(e)
        return 200


if __name__ == "__main__":
    main()
