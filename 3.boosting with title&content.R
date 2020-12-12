require(xgboost)
require(Matrix)
require(data.table)
if (!require('vcd')) install.packages('vcd')

library(vcd)

news_train <- read_csv("news_train.csv")

news = news_train[-which(duplicated(news_train$content)),]
news <- subset(news, select=-date)
news <- subset(news, select=-n_id)
news <- subset(news, select=-ord)

df <- data.table(news, keep.rownames = FALSE)

levels(df[,info])

sparse_matrix <- sparse.model.matrix(info ~ ., data = df)[,-1]
head(sparse_matrix)

output_vector = df[,info] == "0"

bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 10, eta = 0.5, nthread = 2, nrounds = 10, objective = "binary:logistic")
bst
#xbg모델 한번 더 xgboosting
bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 64, eta = 0.05, nthread = 2, nrounds = 10, objective = "binary:logistic",model = bst)

#부스팅 모델 학습에 영향이 큰(importance한) 변수들을 추출한다.
importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
head(importance)

#변수들의 영향률과 순위를 알기 위하여 plot
xgb.plot.importance(importance_matrix = importance)
