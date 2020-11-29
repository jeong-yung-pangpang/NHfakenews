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
importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
head(importance)

xgb.plot.importance(importance_matrix = importance)
