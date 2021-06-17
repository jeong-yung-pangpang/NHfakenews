# 주제 : AI야, 진짜 뉴스를 찾아줘!

(https://dacon.io/competitions/official/235658/overview/description)

팀명 : 정융팡팡(김도연, 안희승, 최수지)

기간 : 2020.11.23 ~ 2021.02.26

## 대회 결과

![dacon 결과](https://user-images.githubusercontent.com/49015100/122323718-7e70d300-cf62-11eb-99f9-a2b6a4ea1620.JPG)

## 데이터셋 설명
- news_train.csv(118,745건)

| No | 컬럼명 | 컬럼설명| 예시
| :---:         |     :---:      | :---:      |         :--- |
| 1   | N_ID    | 뉴스 index 번호|33938
| 2     | DATE       | 뉴스 발행 날짜      |20200518
| 3     | TITLE   | 뉴스 제목      |'디지털 적폐' 공인인증서 퇴출 코앞...20일 본회의 오를 듯
| 4     | CONTENT | 뉴스 내용      |공인인증서가 21년 만에 역사 속으로 사라질 전망이다.
| 5     | ORD     | 뉴스 내용 순서      |1
| 6     | INFO    | 정보유무(1:정보/0:미정보)  |1

- news_text.csv(142,565건)

| No | 컬럼명 | 컬럼설명| 예시
| :---:         |     :---:      | :---:      |         :--- |
| 1   | N_ID    | 뉴스 index 번호|33938
| 2     | DATE       | 뉴스 발행 날짜      |20200518
| 3     | TITLE   | 뉴스 제목      |'디지털 적폐' 공인인증서 퇴출 코앞...20일 본회의 오를 듯
| 4     | CONTENT | 뉴스 내용      |공인인증서가 21년 만에 역사 속으로 사라질 전망이다.
| 5     | ORD     | 뉴스 내용 순서      |1
| 6     | id  | 고유 번호 |NEWS00247_11


## NEWS CONTENT Word Cloud

![wordcloud_top100](https://user-images.githubusercontent.com/49015100/100099109-5163b280-2ea2-11eb-9bab-f3b684dd8950.JPG)

## 모델 설명
R
- XGBoost
- Logistic regression
- Naive Bayes
- Random Forest
- SVM
- bagging
- LSTM

Python
- XGBoost
- Logistic regression
- Naive Bayes
- AutoML
- RNN
- CNN
- BERT
- koBERT
