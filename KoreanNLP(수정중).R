
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

# 차원축소 : 희소성(sparsity)이 0.95가 넘는 열 삭제
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
corTerms[corTerms <= 0.2] <- 0
netTerms <- network(x = corTerms, directed = FALSE)
plot(netTerms, vertex.cex = 1)
set.edge.value(netTerms, attrname = 'edgeSize', value = corTerms * 3)
ggnet2(
  net = netTerms,
  size.min = 3,
  label = TRUE,
  label.size = 3,
  node.size = sna::degree(dat = netTerms),
  edge.size = 'edgeSize',
  family = 'AppleGothic')+
  theme(plot.title = element_text(hjust = 0.5, face = 'bold')
        
  )

# 데이터프레임 만들기
newsSparse = as.data.frame(as.matrix(spdtm))
colnames(newsSparse) = make.names(colnames(newsSparse))

newsSparse$label = as.factor(news$label)

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
cm <- table(pred=pred_class, actual=test$label)
perf_eval(cm)

cm <- table(predictLog, test$label)
perf_eval(cm)
confusionMatrix(table(rf_pred,test$label))

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

#데이터 스플릿
news_train1 <- text[1:32312,]
news_test1 <- text[32313:46161,]

#사전 훈련된 word2vec사용
word2vec_model <- read.word2vec(file = "ko.w2v", nomalize = TRUE)

#조정 가능한 변수, 
max_words <- 15000 #구성할 word_index개수
maxlen <- 32 # 단어의 최대길이

#tokenizer를 이용해서 단어를 정수인코딩 
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(text)
sequences <- texts_to_sequences(tokenizer, nouns3)
word_index1 <- as.matrix(word_index)
word_index <- tokenizer$word_index


#데이터 패딩, 단어의 길이를 맞춤
data = pad_sequences(sequences, maxlen = maxlen)

#데이터 스플릿층
train_matrix = data[1:nrow(news_train1),]
test_matrix = data[(nrow(news_train1)+1):nrow(data),]

labels = news_train_2$info

training_samples = nrow(train_matrix)*0.90
validation_samples = nrow(train_matrix)*0.10

indices = sample(1:nrow(train_matrix))
training_indices = indices[1:training_samples]
validation_indices = indices[(training_samples + 1): (training_samples + validation_samples)]

x_train = train_matrix[training_indices,]
y_train = labels[training_indices]

x_val = train_matrix[validation_indices,]
y_val = labels[validation_indices]

dim(x_train)
table(y_train)

#단어 임베딩
embeddings_index = new.env(hash = TRUE, parent = emptyenv())
news_embedding_dim = 300
news_embedding_matrix = array(0, c(max_words, news_embedding_dim))

#가져는 왔는데 사용 상황자체가 다르기 때문에 큰 의미가 없는듯
for (word in names(word_index)){
  index <- word_index[[word]]
  if(index < max_words){
    news_embedding_vector = embeddings_index[[word]]
    if(!is.null(news_embedding_vector))
      news_embedding_matrix[index+1,] <- news_embedding_vector
  }
}


#input
input <- layer_input(
  shape = list(NULL),
  dtype = "int32",
  name = "input"
)


#hidden layer

embedding <- input %>% 
  layer_embedding(input_dim = max_words, output_dim = news_embedding_dim, name = "embedding")

lstm <- embedding %>% 
  layer_lstm(units = maxlen,dropout = 0.25, recurrent_dropout = 0.25, return_sequences = FALSE, name = "lstm")

dense <- lstm %>%
  layer_dense(units = 128, activation = "relu", name = "dense") 

predictions <- dense %>% 
  layer_dense(units = 1, activation = "sigmoid", name = "predictions")



#모델 구축
model <- keras_model(input, predictions)

# Freeze the embedding weights initially to prevent updates propgating back through and ruining our embedding
# 이거 쓰면 정확도 안나옴, 사용 데이터랑 호환이 안되는 듯..?
# get_layer(model, name = "embedding") %>%
#   set_weights(list(news_embedding1_matrix)) %>%
#   freeze_weights()


# Compile

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = "binary_accuracy"
)

print(model)

#모델 훈련(20분 ~ 30분)
history <- model %>% fit(
  x_train,
  y_train,
  batch_size = 2048,
  validation_data = list(x_val, y_val),
  epochs = 35,
  view_metrics = FALSE,
  verbose = 0
)

#결과 출력
print(history)
plot(history)

#예측값
predictions <- predict(model, test_matrix)
predictions <- ifelse(predictions >= 0.5, 1, 0)

#성능 평가
perf_eval <- function(cm){
  # true positive rate
  TPR = Recall = cm[2,2]/sum(cm[2,])
  # precision
  Precision = cm[2,2]/sum(cm[,2])
  # true negative rate
  TNR = cm[1,1]/sum(cm[1,])
  # accuracy
  ACC = sum(diag(cm)) / sum(cm)
  # balance corrected accuracy (geometric mean)
  BCR = sqrt(TPR*TNR)
  # f1 measure
  F1 = 2 * Recall * Precision / (Recall + Precision)
  
  re <- data.frame(TPR = TPR,
                   Precision = Precision,
                   TNR = TNR,
                   ACC = ACC,
                   BCR = BCR,
                   F1 = F1)
  return(re)
}

cm <- table(pred=predictions, actual=news_test$label)
perf_eval(cm)

