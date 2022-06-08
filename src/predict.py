import autokeras as ak
import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from datasets import load_metric
from pshmodule.utils import filemanager as fm
from tensorflow.keras.models import load_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ElectraForSequenceClassification,
)


@hydra.main(config_name="config.yml")
def main(cfg):
    # electra
    # tokenizer & model
    print(f"load tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)
    print("tokenizer loading done!")

    print(f"load model...", end=" ")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.PATH.save_dir,
        num_labels=cfg.MODEL.num_labels,
    )
    model.eval().cuda()
    print("model loading done!")

    # # predict
    # df = fm.load(cfg.PATH.predict_data_path)
    # print(df.shape)
    # df.dropna(axis=0, inplace=True)
    # print(df.head())
    # print(df.shape)

    sentences = [
        "윤석열 대통령은 7일 문재인 전 대통령의 양산 사저 주변에서 벌어지고 있는 보수단체의 시위에 대해 대통령 집무실도 시위가 허가되는 판이니까 다 법에 따라 되지 않겠나라고 밝혔다 윤 대통령은 이날 오전 용산 대통령 집무실로 출근길에 문재인 전 대통령 양산 사저 시위가 계속되는데 어떻게 보시나라는 취재진의 질문에 이같이 답했다 법과 원칙에 따라 허용 여부를 처리할 문제이지 본인이 개입할 문제는 아니라는 취지로 해석된다",
        "곡물 등 국제 원자재 가격이 오르면서 가공식품 가격이 10년 4개월만에 가장 많이 올랐다 6일 통계청 국가통계포털 KOSIS에 따르면 지난달 외식 물가지수는 1년 전보다 뛰었다. 1998년 3월 이후 24년2개월 만에 가장 높은 상승률을 기록했다",
        "대기업 취업을 미끼로 1억2000만원을 받아 챙긴 60대가 실형을 선고 받았다 울산지법 형사3단독(판사 노서영)은 사기 혐의로 재판에 넘겨진 A씨에게 징역 2년 6개월을 선고했다고 7일 밝혔다 A씨는 2019년 10월 울산의 한 식당에서 지인 B 씨에게 자동차 회사 비서실에 근무한 적이 있는데 아들 2명을 자동차 회사에 취업시켜 줄 수 있다며 1억 2000만원을 받아 챙긴 혐의로 기소됐다",
        "게다가 산불 진화 헬기 상당수가 담수량이 적은데 8천 리터를 담을 수 있는 헬기는 국내 6대에 불과합니다 산이 많은 지형에서 산불에 취약할 수 밖에 없는 취약점을 노출하고 있습니다 이에 경북도가 담수량만 1만 리터 초속 22m의 강풍에도 비행할 수 있는 초대형 헬기 치누크를 도입하기로 했습니다 관련 예산 250억 원을 확보해 2025년 산불 현장에 투입됩니다 특히 야간 진화에 특화된 특수진화대도 조직해 조기 진화에 총력을 기울이기로 했습니다",
        "차량 조수석 문을 열고 여성 운전자를 둔기 등으로 폭행한 남성이 긴급체포됐다 대구강북경찰서는 강도상해 혐의로 40대 남성 A씨를 긴급체포했다고 7일 밝혔다 A씨는 지난 6일 오후 11시 20분쯤 대구시 북구 구암동에서 주차를 막 끝낸 20대 여성 A씨의 차량 조수석 문을 열고 들어가 둔기와 주먹 등으로 B씨를 폭행하며 금품을 뺏으려 한 혐의를 받고 있다 B씨가 극력 저항하자 A씨는 그대로 도주해 금품 탈취는 미수에 그쳤다",
    ]
    df = pd.DataFrame(sentences, columns=["content"])
    print(df.head())

    electra_pred = []
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
            outputs = model(**data)

            predict = np.argmax(outputs.logits[0].cpu().numpy())

            sentiment = ""
            if predict == 0:
                sentiment = "중립"
            elif predict == 1:
                sentiment = "긍정"
            else:
                sentiment = "부정"

        electra_pred.append(sentiment)

    print(electra_pred[:5])
    df["electra"] = electra_pred

    print(df.head())


if __name__ == "__main__":
    main()
