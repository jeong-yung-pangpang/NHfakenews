
install.packages('multilinguer')
library(multilinguer)
install.packages(c("hash", "tau", "Sejong", "RSQLite", "devtools", "bit", "rex", "lazyeval", "htmlwidgets", "crosstalk", "promises", "later", "sessioninfo", "xopen", "bit64", "blob", "DBI", "memoise", "plogr", "covr", "DT", "rcmdcheck", "rversions"), type = "binary")

# KoNLP 설치
install.packages("remotes")
remotes::install_github('haven-jeon/KoNLP', upgrade = "never", INSTALL_opts=c("--no-multiarch"), force = TRUE)
library(KoNLP)
#필요한 패키지 
install.packages('wordcloud')
install.packages('tm')
install.packages('e1071')
install.packages('gmodels')
install.packages('SnowballC')
install.packages('randomForest')
install.packages('dplyr')
install.packages('reshape2')
install.packages('wordcloud2')
install.packages("tidyverse")
install.packages("tidytext")
install.packages("data.table")
install.packages("tensorflow")
install.packages("keras")
install.packages('wordVectors')
install.packages("word2vec")
install.packages("udpipe")
require(tensorflow)
install_tensorflow()
library(wordcloud)
library(tm)
library(e1071)
library(gmodels)
library(SnowballC)
library(randomForest)
library(dplyr)
library(tidyr)
library(reshape2)
library(wordcloud2)
library(keras)
library(tidyverse)
library(tidytext)
library(keras) 
library(data.table) 
library(tensorflow)
library(ggplot2)
library(wordVectors)
library(readr)
library(word2vec)
library(udpipe)
library(caret)

#wordvectors패키지 설치
require(devtools)
devtools::install_github("bmschmidt/wordVectors")
install.packages("wordVectors")
library(wordVectors)

=========================================================================================================================
  
#데이터 전처리 
df_uniq <- unique(news_train$n_id)
length(df_uniq)

# 결측치 확인
sum(is.na(news_train))

# 중복된 content 제거
news_train_2 = news_train[-which(duplicated(news_train$content)),]
text = news_train_2$content

#한글 외 문자 제거
text <- str_replace_all(text, "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
text = gsub("&[[:alnum:]]+;", "", text)            # escape(&amp; &lt;등) 제거
text = gsub("\\s{2,}", " ", text)                  # 2개이상 공백을 한개의 공백으로 처리
text = gsub("[[:punct:]]", "", text)               # 특수문자 제거

# 명사 추출 
nouns <- extractNoun(text)
# nouns <- nouns[nchar(nouns)>=2]  # 단어 길이 2이상 추출 -> 열개수 바뀌어서 모델학습 못함
head(nouns)


# #wordcloud
# df1 <- nouns %>% 
#   mutate(noun=str_match(Value, '([가-힣]+)/N')[,3]) %>%
#   na.omit %>% 
#   filter(str_length(noun)>=2) %>% 
#   count(noun, sort=TRUE) %>%
#   filter(n>=200) %>%
#   wordcloud2(fontFamily='Noto Sans CJK KR Bold', size = 0.5)
# 
# #wordVector를 이용한 word2vec 활용!
# set.seed(1234)
# model <- word2vec(x = nouns3, type = "cbow", dim = 100, iter= 15, encoding = 'utf-8')
# #행렬로 바꾸는 것까지 ok
# embedding <- as.matrix(model)
# embedding[1:5, 1:5]
# #predict쓰니까 한글 깨지고 na값 받아옴..
# embedding1 <- predict(model, c("수협", "항공우주"), type = "embedding")
# lookslike <- predict(model, c("수협", "항공우주"), type = "nearest", top_n = 5)


#말뭉치 형성
corpus <- VCorpus(VectorSource(nouns))

wordcloud2(data = corpus, fontFamily='Noto Sans CJK KR Bold', size = 0.5)

# 단어-문서 행렬(단어 빈도수)
dtm = DocumentTermMatrix(corpus)
dtm
as.matrix(dtm[1:5, 1:20])

# DTM 내에 있는 각 단어에 대한 중요도를 계산할 수 있는 TF-IDF 가중치
# 숫자로 된 단어를 제외하고 2글자 이상 단어만 사용
dtmTfIdf <- DocumentTermMatrix( x = corpus, control = list( removeNumbers = TRUE, wordLengths = c(2, Inf), weighting = function(x) weightTfIdf(x, normalize = TRUE) ))  
dtmTfIdf
as.matrix(dtmTfIdf[1:5, 1:500])

# 차원축소 : 희소성(sparsity)이 0.99가 넘는 열 삭제
spdtm = removeSparseTerms(dtmTfIdf, 0.99)
spdtm
as.matrix(spdtm[1:5, 1:20])

# 키워드 상관 단어 파악
findAssocs(spdtm,'코로나',0.1)

# 상관 행렬 만들기
spdtm %>% as.matrix() %>% cor() -> corTerms
glimpse(corTerms)

# 단어 네트워크맵
install.packages('network')
install.packages('GGally')
install.packages('sna')
library(network)
library(GGally)
library(sna)
corTerms[1:10, 1:10]
dim(corTerms)
netTerms <- network(x = corTerms, directed = FALSE)
plot(netTerms, vertex.cex = 1)

# 데이터프레임 만들기
newsSparse = as.data.frame(as.matrix(spdtm))
colnames(newsSparse) = make.names(colnames(newsSparse))

newsSparse$label = as.factor(news_train_2$label)

# train, test set 분리
library(caTools)
set.seed(123)
spl = sample.split(newsSparse$label, 0.7)
train = subset(newsSparse, spl == TRUE)
test = subset(newsSparse, spl == FALSE)

# 정확도 함수
perf_eval <- function(cm){
  TPR = Recall = cm[2,2]/sum(cm[2,])
  Precision = cm[2,2]/sum(cm[,2])
  TNR = cm[1,1]/sum(cm[1,])
  ACC = sum(diag(cm)) / sum(cm)
  BCR = sqrt(TPR*TNR)
  F1 = 2 * Recall * Precision / (Recall + Precision)
  re <- data.frame(TPR=TPR,
                   Precision = Precision,
                   TNR = TNR,
                   ACC = ACC,
                   BCR = BCR,
                   F1 = F1)
  return(re)
}
                                                           
# 로지스틱 회귀
Logmodel = glm(label ~ ., data=train, family="binomial")
predictLog = predict(Logmodel, newdata=test, type="response") 
table(test$label, predictLog > 0.5)
pred_class<-as.factor(ifelse(predictLog>0.5,1,0))

pred_class <- rep(0, nrow(test))
pred_class[predictLog > 0.5] <- 1
confusionMatrix(table(pred_class, text$label))

# Naive Bayes
library(caret)
naivesubject <- naiveBayes(label~., data=train)
predictnaivesubject <- predict(naivesubject, newdata = test, type="class")
cm <- table(predictnaivesubject, test$label)
perf_eval(cm)
confusionMatrix(table(predictnaivesubject, test$label))

# Random Forest
rf_classifier <- randomForest(label~., data=train, ntree = 300)
rf_classifier
rf_pred = predict(rf_classifier, newdata = test)
cm <- table(rf_pred, test$label)
perf_eval(cm)
confusionMatrix(table(rf_pred,test$label))

# svm
svm_classifier <- svm(label~., data=train)
svm_classifier
svm_pred = predict(svm_classifier,test)
cm <- table(svm_pred, test$label)
perf_eval(cm)
confusionMatrix(svm_pred,test$label)

# bagging
# Averaging
pred_avg <-(as.numeric(as.character(pred_class)) + as.numeric(as.character(rf_pred)) + as.numeric(as.character(svm_pred)))/3
pred_avg <-as.factor(ifelse(pred_avg>0.5,1,0))
cm <- table(pred_avg, test$label)
perf_eval(cm)
confusionMatrix(pred_avg,test$label)
# Majority Voting
pred_majority<-as.factor(ifelse(pred_class==1 & rf_pred==1,1,ifelse(pred_class==1 & svm_pred==1,1,ifelse(rf_pred==1 & svm_pred==1,1,0))))
cm <- table(pred_majority, test$label)
perf_eval(cm)
confusionMatrix(pred_majority,test$label)

==========================================================================================================
test$label <- predictnaivesubject

label_0 <- test %>%
  filter(label == 0)

label_1 <- test %>%
  filter(label == 1)

label_0 <- label_0[1:200,]
label_1 <- label_1[1:200,]

tsne_data <- rbind(label_0, label_1)

tsne_label <- tsne_data[,120]
tsne_data <- tsne_data[,-120]
tsne_data <- as.matrix(tsne_data)

table(tsne_label)
head(tsne_data)

# news 데이터를 matrix로 변환시킨 후 t-SNE 적용
install.packages("tsne")
library("tsne")
news_tsne = tsne::tsne(tsne_data)



