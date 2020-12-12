library(text2vec)
library(xgboost)
library(pdp)

news_train = read_csv('news_train.csv')

news = news_train[-which(duplicated(news_train$content)),]
news <- subset(news, select=-date)
news <- subset(news, select=-n_id)
news <- subset(news, select=-ord)

#content의 문장을 토큰화한다.
vocab <- create_vocabulary(itoken(news$content,
                                  preprocessor = tolower,
                                  tokenizer = word_tokenizer))

# 토큰화된 텍스트를 dtm matrix화한다. 이 코드는 xgboosting에 최적화된 dgCMatrix형태를 반환한다.
dtm_train <- create_dtm(itoken(news$content,
                               preprocessor = tolower,
                               tokenizer = word_tokenizer),
                        vocab_vectorizer(vocab))

train_matrix <- xgb.DMatrix(dtm_train, label = news$info)

# xgboost 모델링, 여기서 eta는 학습률, max_depth는 한 트리의 max_depth, nrounds는 boosting round, objective는 목적 함수이다.
# eta값이 높을수록 과적합이 일어날 수 있으므로 낮게 잡고 nrounds값을 높여 학습률을 조금씩 여러번 높인다.
#xgboost는 초기모델으로, 이 모델을 다시 xgb.cv 메서드를 통해 boosting 가능하다.
xgb_fit <- xgboost(data = train_matrix, eta = 0.01, max_depth = 5, nrounds = 10, objective = "binary:logistic")

set.seed(100)
#
cv <- xgb.cv(data = train_matrix, label = news$info, nfold = 5,
             nrounds = 60, objective = "binary:logistic")

library(caret)
library(Matrix)

# Create our prediction probabilities
pred <- predict(xgb_fit, dtm_train)

# Set our cutoff threshold
pred.resp <- ifelse(pred >= 0.86, 1, 0)
