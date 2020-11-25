if (!require(dplyr))     
  install.packages("dplyr")
if (!require(tidyverse)) 
  install.packages("tidyverse")
if (!require(stringr))   
  install.packages("stringr")
if (!require(memoise))   
  install.packages("memoise")
if (!require(KoNLP))     
  install.packages("KoNLP")
if (!require(wordcloud)) 
  install.packages("wordcloud")
if (!require(extrafont)) 
  install.packages("extrafont")
if (!require(tm))
  install.packages("tm")
if (!require(network))
  install.packages('network')
if (!require(GGally))
  install.packages('GGally')
if (!require(sna))
  install.packages('sna')
if (!require(e1071))
  install.packages("e1071")

install.packages("remotes")
remotes::install_github('haven-jeon/KoNLP', upgrade = "never", INSTALL_opts=c("--no-multiarch"))


# 패키지 로드
library(dplyr)
library(readr)        # 파일 읽기 기능 제공 (tidyverse패키지에 포함됨)
library(stringr)      # 문자열 관련 기능 제공 패키지
library(rJava)        # KoNLP가 의존함 (Java기능 호출 패키지)
library(memoise)      # KoNLP가 의존함
library(KoNLP)        # 한글데이터 형태소 분석 패키지 (이름 대소문자 주의)
library(wordcloud)    # 워드클라우드 생성 패키지
library(RColorBrewer) # 색상 제어 패키지
library(extrafont)    # 폰트관리 패키지
library(randomForest)
library(tm)
library(caret)
library(network)
library(GGally)
library(sna)
library(caTools)
library(e1071)
library(keras)
library(tidyverse) # importing, cleaning, visualising 
library(tidytext) # working with text
library(keras) # deep learning with keras
library(data.table) # fast csv reading
library(tensorflow)
library(ggplot2)
library(multilinguer)
library(stringr)

useNIADic()
# 폰트 스캔
font_import(pattern="NanumGothic.ttf")
loadfonts(device="win")       # Windows
fonts <- fonttable()
unique(fonts$FamilyName)

# 데이터 로드
news_train <- read_csv("news_train.csv")
str(news_train)

# 뉴스 개수
df_uniq <- unique(news_train$n_id)
length(df_uniq)

# 결측치 확인
sum(is.na(news_train))

# 중복된 content 제거
news = news_train[-which(duplicated(news_train$content)),]
news <- subset(news, select=-date)
news <- subset(news, select=-title)
news <- subset(news, select=-n_id)
news <- subset(news, select=-ord)



corpus <- VCorpus(VectorSource(news$content))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stemDocument)

table(news$info)

dtm = DocumentTermMatrix(corpus)
dtm
as.matrix(dtm[1:5, 1:20])

dtmTfIdf <- DocumentTermMatrix( x = corpus, control = list( removeNumbers = TRUE, wordLengths = c(2, Inf), weighting = function(x) weightTfIdf(x, normalize = TRUE) ))  
dtmTfIdf
as.matrix(dtmTfIdf[1:5, 1:500])

spdtm = removeSparseTerms(dtmTfIdf, 0.99)
spdtm

#spdtm %>% as.matrix() %>% cor() -> corTerms
#glimpse(corTerms)

newsSparse = as.data.frame(as.matrix(spdtm))
colnames(newsSparse) = make.names(colnames(newsSparse))
newsSparse$info = as.factor(news$info)


set.seed(150)
spl = sample.split(newsSparse$info, 0.7)
train = subset(newsSparse, spl == TRUE)
test = subset(newsSparse, spl == FALSE)
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

Logmodel = glm(info ~ ., data=train, family="binomial")
predictLog = predict(Logmodel, newdata=test, type="response") 
table(test$info, predictLog > 0.5)
pred_class<-as.factor(ifelse(predictLog>0.5,1,0))

pred_class <- rep(0, nrow(test))
pred_class[predictLog > 0.5] <- 1
cm <- table(pred=pred_class, actual=test$info)
perf_eval(cm)

rf_classifier <- randomForest(info~., data=train, ntree = 300)
rf_classifier
rf_pred = predict(rf_classifier, newdata = test)
cm <- table(rf_pred, test$info)
perf_eval(cm)
confusionMatrix(table(rf_pred,test$info))

svm_classifier <- svm(info~., data=train)
svm_classifier
svm_pred = predict(svm_classifier,test)
cm <- table(svm_pred, test$info)
perf_eval(cm)
confusionMatrix(svm_pred,test$info)
