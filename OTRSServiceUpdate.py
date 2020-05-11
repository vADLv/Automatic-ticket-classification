#!/usr/bin/env python
# coding: utf-8

# # Load the classifier model
# In the [colab python notebook](https://colab.research.google.com/drive/1ddkqWp1YHoxTFaNLUA4KfNTPVUlTP8ZK?usp=sharing) you can see data crawling, preprocessing and model training. From the colab we get saved classifier: *lstm_model.json* and model weights *lstm_model.h5* files with *tokenizer.pickle*.
# Also we need to provide the same preprocessing steps to predict new ticket service.

# In[75]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os
import pickle
import re


# In[73]:


FOLDER_PATH = '/Users/V/Desktop/AIR/p/'
MAX_SEQUENCE_LENGTH = 100


# In[107]:


#read services dict (change 16 -> 0)
services_df = pd.read_excel(FOLDER_PATH+'services.xlsx')
def get_sname(sid):
    if sid==0:
        sid = 16
    return services_df.loc[services_df.id==sid].name.iloc[0]

# loading tokenizer
with open(FOLDER_PATH+'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())

# load json and create model
json_file = open(FOLDER_PATH+'lstm_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(FOLDER_PATH+"lstm_model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])


# In[28]:


import nltk
nltk.download('punkt')

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from itertools import chain

class Preprocessor:

    def __init__(self):
        #english
        #self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                      #'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        self.stop_words = stopwords.words("english")
        self.stem_en = SnowballStemmer("english")

        #russian
        self.stop_words.extend(stopwords.words("russian"))
        self.stem_ru = SnowballStemmer("russian")

        self.signature = ['best regard', 'best regards', 'kind regards','regards', 'yours sincerely',
                          'sincerely yours','yours faithfully','faithfully yours','faithfully','sincerely',
                          'best wishes','с уважением','искренне ваш','с наилучшими пожеланиями','целую']

    def has_cyrillic(self, text):
        return bool(re.search('[\u0400-\u04FF]', text))

    def remove_signature(self, text):
        for r in self.signature:
            if re.search(r,text):
                text = text.split(r)[0]
                break
        return text

    # word tokenize text using nltk lib
    def tokenize(self, text):
        return nltk.word_tokenize(text)

    # stem word using provided stemmer
    def stem(self, word):
        if word[:2]=="//":
            return word.split("/")[2]
        #return word
        if self.has_cyrillic(word):
            return self.stem_ru.stem(word)
        else:
            return self.stem_en.stem(word)

    # check if word is appropriate - not a stop word and isalpha,
    # i.e consists of letters, not punctuation, numbers, dates
    def is_apt_word(self, word):
        #store numbers but delete punctuation
        if word[:2]!="//":
            return word not in self.stop_words and word.isalnum()#and word.isalpha()
        else:
            return True

    # combines all previous methods together
    # tokenizes lowercased text and stems it, ignoring not appropriate words
    def preprocess(self, text):
        tokenized = self.tokenize(self.remove_signature(text.lower()))
        self.stop_words = set(chain(*[nltk.word_tokenize(www) for www in self.stop_words]))
        return ' '.join( [self.stem(w) for w in tokenized if self.is_apt_word(w)] )
        #return ' '.join( [w for w in tokenized if self.is_apt_word(w)] )
        #return ' '.join( [w for w in tokenized if w.isalnum()] )
        #return ' '.join( [self.stem(w) for w in tokenized if w.isalnum()] )


# In[ ]:





# # Demo prediction for one ticket

# In[ ]:


import json
from pyotrs import Article, Client, DynamicField, Ticket
from datetime import datetime
from datetime import timedelta
import time


# In[7]:


with open(FOLDER_PATH+"credentials.json", "r") as f:
    lp = json.load(f)


# In[119]:


client = Client("https://it.innopolis.university", lp['l'], lp['p'])
client.session_create()


# In[16]:


#read ticket by id
client.ticket_get_by_id(6998, articles=1)
my_ticket = client.result[0]
demo_ticket = my_ticket.field_get('Title') + '\n ' + my_ticket.articles[0].field_get('Body')
print(demo_ticket)


# In[108]:


#same preprocessing like in trined model
prep = Preprocessor()
demo_text = prep.preprocess(re.sub(r"R[Ee]: \[Ticket #\d*\]", "", demo_ticket))


# In[109]:


#create dataframe to load into tokenizer
df_demo = pd.DataFrame(data={'demo': [demo_text]})['demo'].astype(str)


# In[110]:


#use pretrained tokenizer
sequence_demo = tokenizer.texts_to_sequences(df_demo)


# In[111]:


index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())
print(" ".join([index_to_word[i] for i in sequence_demo[0]]))
print("||")
print(*sequence_demo[0])


# In[112]:


#predict probabilities, that ticket belongs to service
demo_pred = loaded_model.predict(pad_sequences(sequence_demo, maxlen=MAX_SEQUENCE_LENGTH))


# In[117]:


print(f'Current service: {my_ticket.field_get("Service")}')
print(f'Predicted service: {get_sname(demo_pred.argmax())} - {round(demo_pred.max()*100,2)}%')


# In[ ]:





# # Update OTRS ticket service
# - 1. Connect to OTRS, through customized with *GenericTicketConnectorREST.yml* **web-service** and [**py-otrs**](https://buildmedia.readthedocs.org/media/pdf/pyotrs/latest/pyotrs.pdf) library
# - 2. Read new ticket with empty **service** field
# - 3. **Predict** new service for all retrived tickets
# - 4. **Update** service for all retrived tickets
# - 5. Repeat steps 2-5

# In[210]:


SLEEP_TIME = 60
HOURS = 5
UPDATE = True


# In[216]:


prep = Preprocessor()

print('ticket classifier started...')
while True:
    week = datetime.utcnow() - timedelta(hours=HOURS)
    lc1 = client.ticket_search(TicketCreateTimeNewerDate=week, Services = list(services_df.name))
    lc2 = client.ticket_search(TicketCreateTimeNewerDate=week)
    lc = list(set(lc2)-set(lc1)
    lct = client.ticket_get_by_list(lc),articles=1)
    tickets=[]
    for ticket in lct:
        ticket_text = ticket.field_get('Title') + '\n ' + ticket.articles[0].field_get('Body')
        tickets.append(prep.preprocess(re.sub(r"R[Ee]: \[Ticket #\d*\]", "", ticket_text)))
    df_tickets = pd.DataFrame(data={'text': tickets})['text'].astype(str)
    sequences = tokenizer.texts_to_sequences(df_tickets)
    pred_ticket = loaded_model.predict(pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH))
    for i in range(len(pred_ticket)):
        if UPDATE:
            try:
                client.ticket_update(lc[i], Service=get_sname(pred_ticket[i].argmax()))
                print(f'Service updated for ticket {lc[i]}')
            except:
                print(f'Ticket №{lc[i]} can not be updated')
        print(f'Predicted service for ticket №{lc[i]}: {get_sname(pred_ticket[i].argmax())} - {round(pred_ticket[i].max()*100,2)}%')

    time.sleep(SLEEP_TIME)
