# @Author: Atul Sahay <atul>
# @Date:   2019-09-19T16:22:51+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2019-09-21T21:25:06+05:30



import os

import spacy

# import gui

import re

import csv

import pandas as pd

import heapq

import numpy as np

import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import logging

import matplotlib.pyplot as plt



## READ RAW DATA
## CONVERT CSV TO TSV

#######################################################################################################################
#
# with open(r"dataset\PIAZZA_read.txt", errors='ignore', encoding='utf-8') as tsvfile:  # opening FILE
#     tsv_reader = csv.reader(tsvfile, delimiter="\t")
#     with open(r'dataset\PIAZZA_write.tsv', 'wt', encoding="utf-8") as out_file:
#         tsv_writer = csv.writer(out_file, delimiter='\t')
#         for line in tsv_reader:
#             tsv_writer.writerow(line)
#
#######################################################################################################################


## READ RAW DATA FROM TSV
##CLEAN THE DATA (DATASET SPECIFIC)

#######################################################################################################################
#
# df = pd.read_csv(r"D:\PycharmProjects\piazza_point\dataset\PIAZZA_write.tsv",
#                  sep='\t',
#                  header=0,
#                  skip_blank_lines=True,
#                  error_bad_lines=False,
#                  usecols=[0, 2, 3, 5])
# df = df.dropna()
# df_1 = pd.DataFrame().reindex_like(df)
#
# for ind in df.index:
#     tag_list = df['TAGS'][ind].split(',')
#     if len(tag_list) > 1:
#         if tag_list[1] == 'Planter_bot' or tag_list[1] == 'nex_robotics':
#             df['TAGS'][ind] = tag_list[0]
#         else:
#             df['TAGS'][ind] = np.NaN
#
# df = df.dropna()
# df.to_csv(r"dataset\PIAZZA_single_theme.tsv", sep='\t', index=False)
#
#######################################################################################################################


# READ CLEANED DATA
# PREPROCESS CLEANED DATA

#######################################################################################################################
#
# df = pd.read_csv(r"dataset\PIAZZA_single_theme.tsv",
#                  sep='\t',
#                  header=0
#                  )
#
# df_dict = df.to_dict('index')
#
# nlp = spacy.load('en_core_web_sm')
#
# # Adding Custom Stop Words
# customize_stop_words = ['team', 'id', 'hi', 'dear', 'help', 'problem', 'sir']
# for w in customize_stop_words:
#     nlp.vocab[w].is_stop = True
#
# for value in df_dict.values():
#     doc_h = nlp((re.sub('[ÿÂ()#!0-9=*:?.,+]+', ' ', value['TITLE'])).lower().strip())
#     doc_q = nlp((re.sub('[ÿÂ()#!0-9=*:?.,+]+', ' ', value['MAIN_CONTENT'])).lower().strip())
#     doc_h2 = nlp(" ".join([token.orth_ for token in doc_h if not (token.is_punct or token.is_stop)]))
#     doc_q2 = nlp(" ".join([token.orth_ for token in doc_q if not (token.is_punct or token.is_stop)]))
#     value['TITLE'] = re.sub(' +', ' ', " ".join([token.lemma_ for token in doc_h2 if len(token) > 1]))
#     value['MAIN_CONTENT'] = re.sub(' +', ' ', " ".join([token.lemma_ for token in doc_q2 if len(token) > 1]))
#
# df = pd.DataFrame.from_dict(df_dict, orient='index')
#
# df.to_csv(r"dataset\PIAZZA_clean_single_theme.tsv", sep='\t', index=False)
#
#######################################################################################################################


# READ PREPROCESSED DATA

#######################################################################################################################

def read_file(path):
    df = pd.read_csv(path,
                     sep='\t',
                     header=0
                     )

    df = df.dropna()
    df = df.reset_index(drop = True)
    return df

#######################################################################################################################


#######################################################################################################################
# tags_dict_stoi = {'harvester_bot': 1,
#                   'planter_bot': 2,
#                   'transporter_bot': 3,
#                   }
#
# tags_dict_itos = {1: 'harvester_bot',
#                   2: 'planter_bot',
#                   3: 'transporter_bot'
#                   }
#
# df['LABEL'] = 0
#
# for ind in df.index:
#     df['LABEL'][ind] = tags_dict_stoi[df['TAGS'][ind]]
#
# lab = [1, 2, 3]
#######################################################################################################################


## PLOT CLASS COUNT

#######################################################################################################################
# fig = plt.figure(figsize=(8, 6))
# df.groupby('TAGS').QUERY_ID.count().plot.bar(ylim=0)
# plt.show()
#######################################################################################################################


#######################################################################################################################

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def jaccard_query_similarity(doc_path, query_path):
    df_query = pd.read_csv(query_path,
                     sep='\t',
                     header=0
                     )

    df = read_file(doc_path)

    jac_sim = list()
    head_w = 0.6
    cont_w = 1 - head_w

    # theme = df_query.iloc[0]['TAGS']
    # df_theme = df[df['TAGS'] == theme]
    # df_theme = df_theme.dropna()
    # df_theme = df_theme.reset_index(drop = True)

    # df_size = len(df_theme.index)
    df_size = len(df.index)



    for sl in range(df_size):
        # h_sim = jaccard_similarity(df_query.iloc[0]['TITLE'], df_theme.iloc[sl]['TITLE'])
        # c_sim = jaccard_similarity(df_query.iloc[0]['MAIN_CONTENT'], df_theme.iloc[sl]['MAIN_CONTENT'])
        h_sim = jaccard_similarity(df_query.iloc[0]['TITLE'], df.iloc[sl]['TITLE'])
        c_sim = jaccard_similarity(df_query.iloc[0]['MAIN_CONTENT'], df.iloc[sl]['MAIN_CONTENT'])
        q_w = head_w*h_sim + cont_w*c_sim
        jac_sim.append(q_w)

    # import operator
    # index, value = max(enumerate(jac_sim), key=operator.itemgetter(1))
    # similar_query = df_theme.loc[[index]]

    jac_sim = np.asarray(jac_sim)
    indx = jac_sim.argsort()[-5:][::-1]

    # print("#######################################################################################################")
    # print("JACCARD SIMILARITY")
    # print("#######################################################################################################")
    # print("jaccard Similarity: {}".format(value))
    # print("Similar query:\n", similar_query)
    # print("#######################################################################################################")

    print("#######################################################################################################")
    print("JACCARD SIMILARITY")
    print("#######################################################################################################")
    print("Top 5 Similar queries:\n")
    print(" ID               SUMMARY                      QUERY                            "
          "    THEME                   SIMILARITY")
    for temp, i in enumerate(indx):
        print(df.loc[[i]].to_string(header=False, index=False), jac_sim[indx[temp]])
    print("#######################################################################################################")

#######################################################################################################################


#######################################################################################################################
# doc_l = list()
#
from gensim.models import Word2Vec, KeyedVectors
#
# for ind in df.index:
#     doc_l.append((df.iloc[ind]['TITLE'] + ' ' + df.iloc[ind]['MAIN_CONTENT']).split())
#
#
# model = Word2Vec(doc_l, min_count=1, workers=3, window=3, sg=1)
# print(model.wv.__getitem__('bot'))
#
# model.wv.save_word2vec_format('model.txt', binary=False)
#######################################################################################################################


#######################################################################################################################

def cosine_similarity(spacy_model, query, document):
    doc1 = spacy_model(query)
    doc2 = spacy_model(document)
    return doc1.similarity(doc2)


def cosine_query_similarity(doc_path, query_path):
    df_query = pd.read_csv(query_path,
                           sep='\t',
                           header=0
                           )

    df = read_file(doc_path)

    cos_sim = list()
    headc_w = 0.6
    contc_w = 1 - headc_w

    # theme = df_query.iloc[0]['TAGS']
    # df_theme_c = df[df['TAGS'] == theme]
    # df_theme_c = df_theme_c.dropna()
    # df_theme_c = df_theme_c.reset_index(drop=True)

    # df_size = len(df_theme_c.index)
    df_size = len(df.index)


    nlp_e = spacy.load(r"spacy.word2vec.model")

    for ind in range(df_size):
        # h_sim = cosine_similarity(nlp_e, df_query.iloc[0]['TITLE'], df_theme_c.iloc[ind]['TITLE'])
        # c_sim = cosine_similarity(nlp_e, df_query.iloc[0]['MAIN_CONTENT'], df_theme_c.iloc[ind]['MAIN_CONTENT'])
        h_sim = cosine_similarity(nlp_e, df_query.iloc[0]['TITLE'], df.iloc[ind]['TITLE'])
        c_sim = cosine_similarity(nlp_e, df_query.iloc[0]['MAIN_CONTENT'], df.iloc[ind]['MAIN_CONTENT'])
        q_w = headc_w*h_sim + contc_w*c_sim
        cos_sim.append(q_w)

    # import operator
    # index, value = max(enumerate(cos_sim), key=operator.itemgetter(1))

    cos_sim = np.asarray(cos_sim)
    indx = cos_sim.argsort()[-5:][::-1]

    # print(indx)
    # print(cos_sim[indx])


    print("#######################################################################################################")
    print("COSINE SIMILARITY")
    print("#######################################################################################################")
    print("Top 5 Similar queries:\n")
    print(" ID               SUMMARY                      QUERY                            "
          "    THEME                   SIMILARITY")
    for temp, i in enumerate(indx):
        print(df.loc[[i]].to_string(header=False, index=False), cos_sim[indx[temp]])
    print("#######################################################################################################")

#######################################################################################################################


#######################################################################################################################

# from sklearn.decomposition import PCA
# from matplotlib import pyplot
#
# model = KeyedVectors.load_word2vec_format('D:\PycharmProjects\piazza_point\model.txt')
# X = model[model.wv.vocab]
#
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
#
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()

#######################################################################################################################

if __name__== "__main__":

    # desired_width = 1000
    # pd.set_option('display.width', desired_width)
    # pd.set_option('display.max_columns', 10)
    #
    # app = gui.Post()
    # app.run()

    cosine_query_similarity(
        "dataset\PIAZZA_clean_single_theme.tsv", "query.tsv")

    print("\n")

    jaccard_query_similarity(
        "PIAZZA_clean_single_theme.tsv", "query.tsv")
