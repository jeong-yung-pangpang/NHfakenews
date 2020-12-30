# -*- coding: utf-8 -*-
"""koBERT_with pytorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nek28-GD3-4SkL4l3kIWDtt2Y2GH4xJj
"""

!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers==3
!pip install torch
!pip install konlpy

!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

##GPU 사용 시
device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model()

from google.colab import drive
drive.mount('/content/drive')

import csv
trainset_csv = '/content/drive/My Drive/news_train.csv'

import pandas as pd
df = pd.read_csv(trainset_csv)
df.head()

df_del = df.drop_duplicates(['content'], keep='first')
#df_del = df
df_del_all = df_del.drop(['n_id','title','date','ord'], axis=1)
df_del_all['content'] = df_del_all['content'].str.replace(pat=r'[^가-힣A-Za-z]', repl= r' ', regex=True)  # replace all special symbols to space

df_del_all.head()

df_del_all.to_csv(path_or_buf='/content/drive/My Drive/news_tr.csv',sep='\t',line_terminator='\r', encoding = 'UTF-8')

ts = '/content/drive/My Drive/news_tr.csv'
df_2 = pd.read_csv(ts, sep='\t',index_col=0)

len(df_2)

from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(df_2, test_size=0.1, random_state=42)
#trainset = df_2.iloc[:41545]
trainset.to_csv(path_or_buf='/content/drive/My Drive/trainset.csv',sep='\t',line_terminator='\r', encoding = 'UTF-8')
#testset = df_2.iloc[41545:]
testset.to_csv(path_or_buf='/content/drive/My Drive/testset.csv',sep='\t',line_terminator='\r', encoding = 'UTF-8')

trainset_txt = '/content/drive/My Drive/news_train.txt'
testset_txt = '/content/drive/My Drive/news_test.txt'
with open(trainset_txt, "w+") as my_output_file:
    with open('/content/drive/My Drive/trainset.csv', "r") as my_input_file:
        [my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
with open(testset_txt, "w+") as my_output_file:
    with open('/content/drive/My Drive/testset.csv', "r") as my_input_file:
        [my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

dataset_train = nlp.data.TSVDataset(trainset_txt, field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset(testset_txt, field_indices=[1,2], num_discard_samples=1)

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## 파라미터 세팅
max_len = 128
batch_size = 64
warmup_ratio = 0.01
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

####주의####
##전처리 안하고 집어넣으면 'invalid literal for int() with base 10' 오류 발생
##전처리 먼저 하고 진행할 것
data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

PATH = '/content/drive/My Drive/'
torch.save(model.state_dict(), PATH + 'model.pt')
#model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수

"""##예측 코드"""

import pandas as pd
testset_csv =  '/content/drive/My Drive/news_test.csv'
news = pd.read_csv(testset_csv,encoding = 'utf-8')

import time
start = time.time()

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device('cuda:0')
bertmodel, vocab = get_pytorch_kobert_model()

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

model.load_state_dict(torch.load("/content/drive/My Drive/model.pt"))

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

news = news.drop(['n_id','title','date','ord'], axis=1)
news["content"] = news["content"].apply(lambda x : ' '.join(x.strip().split()))
news["info"] = 0
news[["content","info"]].to_csv("/content/drive/My Drive/news_test.txt", sep='\t', index=False)
dataset_test = nlp.data.TSVDataset("/content/drive/My Drive/news_test.txt", field_indices=[0,1], num_discard_samples=1)
data_test = BERTDataset(dataset_test, 0, 1, tok, 128, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=64, num_workers=5)

model.eval()

Predict = []

for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)
    _, predict = torch.max(out,1)
    Predict.extend(predict.tolist())

news["info"] = Predict
print(news.groupby("info").size())
news = news[["id","info"]]

print(time.time() - start)

news.to_csv('/content/drive/My Drive/sample_submission.csv', index=False)