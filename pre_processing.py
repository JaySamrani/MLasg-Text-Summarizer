import numpy as np 
import pandas as pd 
import os
import re
from time import time
import spacy
from tqdm import tqdm 
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

label = pd.read_csv('/kaggle/input/news-summary/news_summary.csv', encoding='iso-8859-1')
para = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv', encoding='iso-8859-1')

para_pre2 = label.iloc[:,0:6].copy()
para_pre1 = para.iloc[:,0:2].copy()
para_pre2['text'] = para_pre2['text'].str.cat(para_pre2['ctext'], sep = " ")

data = pd.DataFrame()
data['text'] = pd.concat([para_pre1['text'], para_pre2['text']], ignore_index=True)
data['label'] = pd.concat([para_pre1['headlines'],para_pre2['headlines']],ignore_index = True)

def strip(column):
    for row in column:
        cleaning_patterns = [
            (r"(\\t|\\r|\\n)", ' '),  # Remove escape characters
            (r"(__+|-+|~+|\+\+|\.\.+)", ' '),  # Remove consecutive special characters
            (r"[<>()|&©ø\[\]\'\",;?~*!]", ' '),  # Remove specific special characters
            (r"mailto:", ' '),  # Remove "mailto:"
            (r"\\x9\d", ' '),  # Remove \x9* in text
            (r"([iI][nN][cC]\d+)", 'INC_NUM'),  # Replace INC nums with INC_NUM
            (r"([cC][mM]\d+|[cC][hH][gG]\d+)", 'CM_NUM'),  # Replace CM# and CHG# with CM_NUM
            (r"(\.\s+|\-\s+|\:\s+)", ' '),  # Remove specific punctuation at the end of words
            (r"(https*:\/*)([^\/\s]+)(.[^\s]+)", r'\2')  # Replace URLs with domain
        ]

        for pattern, replacement in cleaning_patterns:
            row = re.sub(pattern, replacement, row)

        row = re.sub(r"\s+", ' ', row).strip()  # Remove multiple spaces and trim

        yield row


brief_cleaning1 = strip(data['text'])
brief_cleaning2 = strip(data['label'])

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

#Taking advantage of spaCy .pipe() method to speed-up the cleaning process:
#If data loss seems to be happening(i.e len(text) = 50 instead of 75 etc etc) in this cell , decrease the batch_size parametre 
t = time()

#Batch the data points into 5000 and run on all cores for faster preprocessing
text = [str(doc) for doc in tqdm(nlp.pipe(brief_cleaning1, batch_size=5000))]

#Takes 7-8 mins
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

#Taking advantage of spaCy .pipe() method to speed-up the cleaning process:
t = time()

#Batch the data points into 5000 and run on all cores for faster preprocessing
label = ['_START_ '+ str(doc) + ' _END_' for doc in tqdm(nlp.pipe(brief_cleaning2, batch_size=5000))]

#Takes 7-8 mins
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

data['cleaned_text'] = pd.Series(text)
data['cleaned_label'] = pd.Series(label)

text_count = []
label_count = []

for sent in data['cleaned_text']:
    text_count.append(len(sent.split()))
for sent in data['cleaned_label']:
    label_count.append(len(str(sent).split()))

graph_df= pd.DataFrame()
graph_df['text']=text_count
graph_df['label']=label_count

graph_df.hist(bins = 5)
plt.show()


#Check how much % of label have 0-15 words
cnt=0
for i in data['cleaned_label']:
    if(len(str(i).split())<=15):
        cnt=cnt+1
print(cnt/len(data['cleaned_label']))

#Check how much % of text have 0-70 words
cnt=0
for i in data['cleaned_text']:
    if(len(i.split())<=100):
        cnt=cnt+1
print(cnt/len(data['cleaned_text']))

#Model to summarize the text between 0-15 words for label and 0-100 words for Text
max_text_len=100
max_label_len=15

#Select the Summaries and Text between max len defined above

cleaned_text =np.array(data['cleaned_text'])
cleaned_label=np.array(data['cleaned_label'])

short_text=[]
short_label=[]

for i in range(len(cleaned_text)):
    if(len(str(cleaned_label[i]).split())<=max_label_len and len(str(cleaned_text[i]).split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_label.append(cleaned_label[i])
        
post_pre=pd.DataFrame({'text':short_text,'label':short_label})

post_pre['label'] = post_pre['label'].fillna('')  # Replace NaN values with an empty string
post_pre['label'] = post_pre['label'].apply(lambda x: 'sostok ' + str(x) + ' eostok')


x_tr,x_val,y_tr,y_val=train_test_split(np.array(post_pre['text']),np.array(post_pre['label']),test_size=0.1,random_state=0,shuffle=True)

#Lets tokenize the text to get the vocab count , you can use Spacy here also
#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))

thresh=4

cnt=0
tot_cnt=0
freq=0
tot_freq=0

for key,value in x_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    if(value<thresh):
        cnt=cnt+1
        freq=freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)



#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences (i.e one-hot encodeing all the words)
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1

print("Size of vocabulary in X = {}".format(x_voc))

#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

thresh=6

cnt=0
tot_cnt=0
freq=0
tot_freq=0

for key,value in y_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    if(value<thresh):
        cnt=cnt+1
        freq=freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
y_tokenizer.fit_on_texts(list(y_tr))

#convert text sequences into integer sequences (i.e one hot encode the text in Y)
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=max_label_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_label_len, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1
print("Size of vocabulary in Y = {}".format(y_voc))

ind=[]
for i in range(len(y_tr)):
    cnt=0
    for j in y_tr[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_tr=np.delete(y_tr,ind, axis=0)
x_tr=np.delete(x_tr,ind, axis=0)

ind=[]
for i in range(len(y_val)):
    cnt=0
    for j in y_val[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_val=np.delete(y_val,ind, axis=0)
x_val=np.delete(x_val,ind, axis=0)

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
