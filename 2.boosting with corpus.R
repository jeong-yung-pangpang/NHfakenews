library(text2vec)
library(xgboost)
library(pdp)

#stringsAsFactors ??
news_train = read_csv('news_train.csv')

news = news_train[-which(duplicated(news_train$content)),]
news <- subset(news, select=-date)
news <- subset(news, select=-n_id)
news <- subset(news, select=-ord)

vocab <- create_vocabulary(itoken(news$content,
                                  preprocessor = tolower,
                                  tokenizer = word_tokenizer))

# Build a document-term matrix using the tokenized review text. This returns a dgCMatrix object
dtm_train <- create_dtm(itoken(news$content,
                               preprocessor = tolower,
                               tokenizer = word_tokenizer),
                        vocab_vectorizer(vocab))

train_matrix <- xgb.DMatrix(dtm_train, label = news$info)

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
