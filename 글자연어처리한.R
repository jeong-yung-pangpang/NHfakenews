# java, rJava 설치 install.packages("multilinguer")
# 이때 mac 사용자는 데스크탑 비밀번호를 물어봅니다. 입력해줘야 설치가 진행됩니다.
install.packages('multilinguer')
library(multilinguer)
install_jdk()
# 위 함수에서 에러가 발생하면 알려주세요
# https://github.com/mrchypark/multilinguer/issues
# 의존성 패키지 설치

install.packages(c("hash", "tau", "Sejong", "RSQLite", "devtools", "bit", "rex", "lazyeval", "htmlwidgets", "crosstalk", "promises", "later", "sessioninfo", "xopen", "bit64", "blob", "DBI", "memoise", "plogr", "covr", "DT", "rcmdcheck", "rversions"), type = "binary")

# github 버전 설치
install.packages("remotes")
# 64bit 에서만 동작합니다.
remotes::install_github('haven-jeon/KoNLP', upgrade = "never", INSTALL_opts=c("--no-multiarch"), force = TRUE)

install.packages("rJava")
source("https://install-gith
library(rJava)
library(KoNLP)

ub.me/talgalili/installr")
installr::install.java()
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
library(tidyverse) # importing, cleaning, visualising 
library(tidytext) # working with text
library(keras) # deep learning with keras
library(data.table) # fast csv reading
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
text = gsub("[[:punct:]]", "", text)               # 특수 문자 제거 (앞의 처리 때문에 마지막에 해야 

mp <- SimplePos09(text)
mp

#형태소 분석()
m_df <- mp%>%melt%>%as_tibble
m_df_copy <- m_df[,c(3,1)]
m_df_copy

#Simplepos09로 변호
m_df_copy_1 <- m_df_copy %>% 
  mutate(noun=str_match(value, '([가-힣]+)/N')[,2]) %>%
  na.omit %>% 
  filter(str_length(noun)>=2) %>% 
  count(noun, sort=TRUE) 


# 명사 추출 
nouns <- sapply(text,extractNoun,USE.NAMES = F) #인코딩문제 있어서 sapply이용해서 저장해봤는데 의미없음
Encoding(nouns3) = 'UTF-8' #tokenizer에서 한글꺠져서 시도했는데 변화없음
nouns <- extractNoun(text)
nouns <- nouns[nchar(nouns)>=2]  # 단어 길이 2이상 추출 
nouns2 <- as.data.frame(as.matrix(nouns)) # 이걸로는 
nouns3 <- as.character(nouns) # tokenizer에 적용하려면 char로 바꿔야함
head(nouns)


#wordVector를 이용한 word2vec 활용!
set.seed(1234)
model <- word2vec(x = nouns3, type = "cbow", dim = 100, iter= 15, encoding = 'utf-8')
#행렬로 바꾸는 것까지 ok
embedding <- as.matrix(model)
embedding[1:5, 1:5]
#predict쓰니까 한글 깨지고 na값 받아옴..
embedding1 <- predict(model, c("수협", "항공우주"), type = "embedding")
lookslike <- predict(model, c("수협", "항공우주"), type = "nearest", top_n = 5)

nrow(nouns2)

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


