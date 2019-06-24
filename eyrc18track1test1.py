import pandas as pd
import numpy as np

import csv
import time

import spacy
from spacy.lang.en import English
import en_core_web_sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline


def feature_map(word):
    '''Simple feature map.'''
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),
                     word.isdigit(),  word.isalpha()])

def feature_map2(word):
    '''Simple feature map.'''
    #print(len(word))
    fvec = []
    for w in word:
        fvec.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w),
                     w.isdigit(),  w.isalpha()]).astype(np.float))
    return fvec

class FeatureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.memory_tagger = MemoryTagger()
        self.tag_encoder = LabelEncoder()
        self.pos_encoder = LabelEncoder()
        
    def fit(self, X, y):
        words = X["Word"].values.tolist()
        self.pos = X["POS"].values.tolist()
        tags = X["Tag"].values.tolist()
        self.memory_tagger.fit(words, tags)
        self.tag_encoder.fit(tags)
        self.pos_encoder.fit(self.pos)
        return self
    
    def transform(self, X, y=None):
        def pos_default(p):
            if p in self.pos:
                return self.pos_encoder.transform([p])[0]
            else:
                return -1
        
        pos = X["POS"].values.tolist()
        words = X["Word"].values.tolist()
        out = []
        for i in range(len(words)):
            w = words[i]
            p = pos[i]
            if i < len(words) - 1:
                wp = self.tag_encoder.transform(self.memory_tagger.predict([words[i+1]]))[0]
                posp = pos_default(pos[i+1])
            else:
                wp = self.tag_encoder.transform([y])[0]
                posp = pos_default(".")
            if i > 0:
                if words[i-1] != ".":
                    wm = self.tag_encoder.transform(self.memory_tagger.predict([words[i-1]]))[0]
                    posm = pos_default(pos[i-1])
                else:
                    wm = self.tag_encoder.transform([y])[0]
                    posm = pos_default(".")
            else:
                posm = pos_default(".")
                wm = self.tag_encoder.transform([y])[0]
            out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                                 self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
                                 pos_default(p), wp, wm, posp, posm]))
        return out


class MemoryTagger(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        '''
        Expects a list of words as X and a list of tags as y.
        '''
        voc = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t: 1}
        self.memory = {}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)
    
    def predict(self, X, y=None):
        '''
        Predict the the tag from memory. If word is unknown, predict 'O'.
        '''
        return [self.memory.get(x, str(y)) for x in X]


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
    
    def get_next(self):
        try:
            s = self.data[self.data["ID"] == (self.n_sent+14)]
            self.n_sent += 1
            return s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist()    
        except:
            self.empty = True
            return None, None, None
        

data = pd.read_csv("F:/World of Smita/PiazzaVirtualAssistant-master/PiazzaVirtualAssistant-master/mydata.csv", encoding="latin1")

#print(data)

data = data.fillna(method="ffill")

print(len(data))

#print("\n")
#print(data.tail(10))

nlp = English()

nlp2 = spacy.load("en_core_web_sm")

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
#print(spacy_stopwords)

#customize_stop_words = ['pin', 'Dear', 'Respected',')','(']

#for w in customize_stop_words: 
#  nlp.vocab[w].is_stop = True

sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)


text = data["MAIN_CONTENT"].values
print(text[14])

tgs = data["TAGS"].values

doc = []
doc2 = []

words = []
tags = []
pos = []
nlp = spacy.load("en_core_web_sm")


csvData = ""
with open('data-eYRC18TRack1.csv',  'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    csvData = [['ID', 'Word', 'POS', 'Tag']]
    for i in range(0,len(text)):
        #writer.writerows([str(i),"","",""])
        doc = nlp(u" " + str(text[i]))
        for token in doc:
            if(token.is_stop == False or token.is_punct==False):
                #print(token.text, token.pos_, token.dep_)
                #writer.writerows([' ', u""+token.text, u""+token.pos_,u""+token.tag_])
                doc2.append({"id":i, "Words":token.text, "POS":token.pos_, "TAG":tgs[i]})
                words.append(token.text)
                pos.append(token.pos_)
                tags.append(token.tag_)
                if(token.pos_ is not 'SPACE'):
                    csvData +=[[str(i),token.text,token.pos_,tgs[i]],]
                else:
                    continue
    writer.writerows(csvData)
               
writeFile.close()

time.sleep(2)


data2 = pd.read_csv("F:/World of Smita/PiazzaVirtualAssistant-master/PiazzaVirtualAssistant-master/data-eYRC18TRack1.csv", encoding="latin1")

#print(data)

data2 = data2.fillna(method="ffill")

wd = data2["Word"].values.tolist()
tg = data2["Tag"].values.tolist()

getter = SentenceGetter(data2)

sent2, pos2, tag2 = getter.get_next()

#print(sent2); print(pos2); print(tag2)

tagger = MemoryTagger()

tagger.fit(sent2, tag2)

#print(tagger.predict(sent2))

tg2 = tagger.tags
print(tg2)

pred = cross_val_predict(estimator=MemoryTagger(), X=wd, y=tg, cv=5)

report = classification_report(y_pred=pred, y_true=tg)
print(report)

words3 = [feature_map(str(w)) for w in wd]
words4 = [feature_map2(str(w)) for w in wd[1:]]

pred2 = cross_val_predict(RandomForestClassifier(n_estimators=20),
                         X=words3, y=tg, cv=5)

report2 = classification_report(y_pred=pred2, y_true=tg)
print(report2)
