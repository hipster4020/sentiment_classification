# 🤖 News Sentiment Classification
## 👉🏻 model
Electra with Active Learning
<br>
<br>

### sentiment : 긍정, 중립, 부정
## 👉🏻 설명
고정된 양의 데이터가 주어지면 semi-supervise 또는 unsupervised learning의 성능은 full-supervised learning에 제한됩니다.<br>
또한, 주석 비용은 대상 작업에 따라 크게 다릅니다.<br><br>
image.png<br>
연결된 loss prediction module은 label이 없는 입력에서 loss 값을 예측합니다.<br>
label이 지정되지 않은 pool의 모든 data point는 loss prediction module에 의해 evaluate 됩니다.<br><br>
image.png
1. Electra Model에 dataset를 input하여 attenttion block 12에 대한 outputs에서 hidden_states를 받아 해당 hidden_states의 값을 LPM(Loss Predict Module)의 입력으로 넣습니다.<br>
2. LPM 내부에서 각각 들어온 input에 대해 Adaptive Average Pooling → flatten → FC layer dense →  ReLU를 수행하고 각각의 층을 concat하여 하나의 layer로 추출합니다.<br>
3. 1에서 나온 Electra Model outputs.logits과 labels 데이터로 추출한 CrossEntropyLoss와 LPM에서 추출된 predict loss를 더해 최종 loss로 사용합니다.<br>
<br>

## 👉🏻 요약
label annotation 추가 작업 시에 많은 공수와 비용이 발생하므로 적절한 Active Learining 사용으로 모델의 효율의 크게 증가할 수 있을 것으로 기대됩니다.<br>
<br>

## 👉🏻 참고 자료

<blockquote>논문<br>
https://arxiv.org/abs/1905.03677v1](https://arxiv.org/abs/1905.03677v1<br>
깃허브<br>
https://github.com/seominseok0429/Learning-Loss-for-Active-Learning-Pytorch/blob/master/main.py
</blockquote>
<br>
<br>
<br>

## 👉🏻 tree
  * [src]
    * [dataloader.py]
    * [test.py]
    * [tokenizer.py]
    * [train.py]
  * [Dockerfile]
  * [README.md]
  * [requirements.txt]