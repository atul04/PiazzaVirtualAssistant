###### Reference Blog: https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/ ######################################



import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def feature_map(word):
    '''Simple feature map.'''
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),
                     word.isdigit(),  word.isalpha()])

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
                wp = self.tag_encoder.transform(['O'])[0]
                posp = pos_default(".")
            if i > 0:
                if words[i-1] != ".":
                    wm = self.tag_encoder.transform(self.memory_tagger.predict([words[i-1]]))[0]
                    posm = pos_default(pos[i-1])
                else:
                    wm = self.tag_encoder.transform(['O'])[0]
                    posm = pos_default(".")
            else:
                posm = pos_default(".")
                wm = self.tag_encoder.transform(['O'])[0]
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
        return [self.memory.get(x, 'O') for x in X]

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
    
    def get_next(self):
        try:
            s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist()    
        except:
            self.empty = True
            return None, None, None

data = pd.read_csv("F:/World of Smita/PiazzaVirtualAssistant-master/PiazzaVirtualAssistant-master/ner_dataset/ner_dataset.csv", encoding="latin1")

#print(data)

data = data.fillna(method="ffill")

#print("\n")
#print(data.tail(10))

words = list(set(data["Word"].values))
#print(words)

n_words = len(words)

#print(n_words)

getter = SentenceGetter(data)

sent, pos, tag = getter.get_next()


###############Data Prepared#######################################################################

#print(sent)

tagger = MemoryTagger()

tagger.fit(sent, tag)

#print(tagger.predict(sent))

#print(tagger.tags)

words2 = data["Word"].values.tolist()

tags2 = data["Tag"].values.tolist()

pred = cross_val_predict(estimator=MemoryTagger(), X=words2, y=tags2, cv=5)

report = classification_report(y_pred=pred, y_true=tags2)
#print(report)

###############Memory Based Classification Accuracy measured#####################################

words3 = [feature_map(w) for w in data["Word"].values.tolist()]

pred2 = cross_val_predict(RandomForestClassifier(n_estimators=20),
                         X=words3, y=tags2, cv=5)

report = classification_report(y_pred=pred2, y_true=tags2)
#print(report)

###############Random Forest Classification Accuracy measured####################################

pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()), 
                                   ("clf", RandomForestClassifier(n_estimators=20, n_jobs=3))]),
                         X=data, y=tags2, cv=5)

report = classification_report(y_pred=pred3, y_true=tags2)
print(report)
