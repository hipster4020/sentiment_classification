# ๐ค News Sentiment Classification
## ๐๐ป model
Electra with Active Learning<br><br>
pretrained model "monologg/koelectra-small-v2-discriminator"์ ํ์ฉํ์ฌ Downstream Task๋ก ํ์ตํ ๋ชจ๋ธ๋ก active learning์ ๊ตฌํ ๋ฐ label annotation ์ถ๊ฐํจ<br>
<br>
<br>

### sentiment : ๊ธ์ , ์ค๋ฆฝ, ๋ถ์ 
<br><br>
## ๐๐ปย ์ค๋ช
๊ณ ์ ๋ ์์ ๋ฐ์ดํฐ๊ฐ ์ฃผ์ด์ง๋ฉด semi-supervise ๋๋ unsupervised learning์ ์ฑ๋ฅ์ full-supervised learning์ ์ ํ๋ฉ๋๋ค.<br>
๋ํ, ์ฃผ์ ๋น์ฉ์ ๋์ ์์์ ๋ฐ๋ผ ํฌ๊ฒ ๋ค๋ฆ๋๋ค.<br><br><br><br><br>
![image](https://user-images.githubusercontent.com/26425581/172622215-09c3748f-c0b1-4a2a-aac4-a0d8aeadad34.png)<br>
์ฐ๊ฒฐ๋ loss prediction module์ label์ด ์๋ ์๋ ฅ์์ loss ๊ฐ์ ์์ธกํฉ๋๋ค.<br>
label์ด ์ง์ ๋์ง ์์ pool์ ๋ชจ๋  data point๋ loss prediction module์ ์ํด evaluate ๋ฉ๋๋ค.<br><br><br><br><br>
![image](https://user-images.githubusercontent.com/26425581/172622399-c847ff0d-d3ac-4d63-badd-9c140d1abded.png)<br>
1. Electra Model์ dataset๋ฅผ inputํ์ฌ attenttion block 12์ ๋ํ outputs์์ hidden_states๋ฅผ ๋ฐ์ ํด๋น hidden_states์ ๊ฐ์ LPM(Loss Predict Module)์ ์๋ ฅ์ผ๋ก ๋ฃ์ต๋๋ค.<br>
2. LPM ๋ด๋ถ์์ ๊ฐ๊ฐ ๋ค์ด์จ input์ ๋ํด Adaptive Average Pooling โ flatten โ FC layer dense โ  ReLU๋ฅผ ์ํํ๊ณ  ๊ฐ๊ฐ์ ์ธต์ concatํ์ฌ ํ๋์ layer๋ก ์ถ์ถํฉ๋๋ค.<br>
3. 1์์ ๋์จ Electra Model outputs.logits๊ณผ labels ๋ฐ์ดํฐ๋ก ์ถ์ถํ CrossEntropyLoss์ LPM์์ ์ถ์ถ๋ predict loss๋ฅผ ๋ํด ์ต์ข loss๋ก ์ฌ์ฉํฉ๋๋ค.<br>
<br>

## ๐๐ปย ์์ฝ
label annotation ์ถ๊ฐ ์์ ์์ ๋ง์ ๊ณต์์ ๋น์ฉ์ด ๋ฐ์ํ๋ฏ๋ก ์ ์ ํ Active Learining ์ฌ์ฉ์ผ๋ก ๋ชจ๋ธ์ ํจ์จ์ ํฌ๊ฒ ์ฆ๊ฐํ  ์ ์์ ๊ฒ์ผ๋ก ๊ธฐ๋๋ฉ๋๋ค.<br>
<br>

## ๐๐ปย macro f1score
<img width="580" alt="1" src="https://user-images.githubusercontent.com/26425581/181575317-f6f71fdc-3c54-44c3-afbc-c262ca07d8f0.png">
<br>
<br>

## ๐๐ปย ์ฐธ๊ณ  ์๋ฃ

<blockquote>๋ผ๋ฌธ<br>
https://arxiv.org/abs/1905.03677v1</blockquote>
<blockquote>๊นํ๋ธ<br>
https://github.com/seominseok0429/Learning-Loss-for-Active-Learning-Pytorch/blob/master/main.py</blockquote>
<br>
<br>
<br>

## ๐๐ป tree
  * [src]
    * [dataloader.py]
    * [test.py]
    * [tokenizer.py]
    * [train.py]
  * [Dockerfile]
  * [README.md]
  * [requirements.txt]