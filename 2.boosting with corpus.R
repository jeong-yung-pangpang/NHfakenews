library(text2vec)
library(xgboost)
library(pdp)

#stringsAsFactors ??
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

# xgboost 모델링, 여기서 eta는 , max_depth는 , nrounds는 , objective는 이다.
xgb_fit <- xgboost(data = train_matrix, eta = 0.01, max_depth = 5, nrounds = 10, objective = "binary:logistic")

set.seed(100)
cv <- xgb.cv(data = train_matrix, label = news$info, nfold = 5,
             nrounds = 60, objective = "binary:logistic")

library(caret)
library(Matrix)

# Create our prediction probabilities
pred <- predict(xgb_fit, dtm_train)

# Set our cutoff threshold
pred.resp <- ifelse(pred >= 0.86, 1, 0)

# Create the confusion matrix
