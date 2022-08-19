from konlpy.tag import Hannanum
import re
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#stopwords = pd.read_table("C:/python/mrc_demomake/tools/korean100.txt", sep="\t")
#stopwords = stopwords.iloc[:,0].tolist()
docs = pd.read_pickle("/Users/yul/Desktop/기업프로젝트/wisenut_demo/tools/inverted_index.pickle")

def __init__(self, query, string, steps, n, dataframe, string_list, tfidf, returned_docs_df):
    self.query = query
    self.string = string
    self.steps = steps
    self.n = n
    self.dataframe = dataframe
    self.string_list = string_list
    self.tfidf = tfidf
    self.returned_docs_df = returned_docs_df
# 품사 태깅
def query_tagger(query):
    q = Hannanum().pos(query)
    string_li = [q[i][0] for i in range(len(q))]
    for i in range(len(q)): # 일반명사, 고유명사, 수사, 숫자, 형용사, 동사 'NC', 'NQ', 'NN',
        if q[i][1] not in ('N', 'P'): #'PA', 'PV'):
            string_li.remove(q[i][0])
    return string_li

def n_gram(string, n):
    string = re.sub(r"\.", "", string)
    li2=[]
    li = string.split(" ")
    for i in li:
        for j in range(len(i)-n):
            li2.append(i[j:j+n])
        for j in range(n):
            li2.append(i[-1-j:])
        li2 = list(set(li2))
    return li2

# def tf_idf_score(dataframe, string_list):
#    doc_freq = pd.Series()
#    for i in string_list:
#        doc_freq = pd.concat([doc_freq, dataframe.iloc[:,0].map(lambda x : str(x).count(i)).rename(f"{i}")], axis = 1)
#        #tf
#    a = doc_freq.iloc[:,1:]
#    a[a > 0] = 1
#    idoc_freq = np.log(a*len(a.index)/(a.sum()+1)) # idf
#    idoc_freq[idoc_freq < 0], idoc_freq[idoc_freq == np.inf] = 0, 0
#    return idoc_freq * doc_freq.iloc[:,1:]

def bm25(dataframe, string_list):
    k1 = 1.2
    b = 0.75
    docLength = 10
    avgDocLength = 30
    doc_freq = pd.Series()
    for i in string_list:
        doc_freq = pd.concat([doc_freq, dataframe.iloc[:,0].map(lambda x : str(x).count(i)).rename(f"{i}")], axis = 1)
        bm25_TF = doc_freq / (doc_freq + k1 * (1-b+b*docLength/avgDocLength))  # bm25_tf
    a = doc_freq.iloc[:,1:]
    a[a > 0] = 1
    idoc_freq = np.log(1 + (a*len(a.index) - a.sum()+0.5) / (a.sum()+0.5)) # bm25_idf
    idoc_freq[idoc_freq < 0], idoc_freq[idoc_freq == np.inf] = 0, 0
    bm25_result = idoc_freq * bm25_TF.iloc[:,1:]
    bm25_result = bm25_result.sum(axis=1).sort_values(ascending=False).head().index.tolist()
    top_1 = dataframe.iloc[bm25_result[0], 0]
    return top_1

def tf_idf_score(dataframe, string_list):
    doc_freq = pd.Series()
    for i in string_list:
        doc_freq = pd.concat([doc_freq, dataframe.iloc[:,0].map(lambda x : str(x).count(i)).rename(f"{i}")], axis = 1)
        #tf
    a = doc_freq.iloc[:,1:]
    a[a > 0] = 1
    idoc_freq = np.log(a*len(a.index)/(a.sum()+1)) # idf
    idoc_freq[idoc_freq < 0], idoc_freq[idoc_freq == np.inf] = 0, 0
    tfidf_result = idoc_freq * doc_freq.iloc[:,1:]
    tfidf_result = tfidf_result.sum(axis=1).sort_values(ascending=False).head().index.tolist()
    top_1 = dataframe.iloc[tfidf_result[0], 0]
    return top_1
# 코사인 유사도 비교
def get_tf_idf_query_similarity(tfidf, returned_docs_df, query):
    docs_list = list(docs.iloc[:, 0])
    docs_tfidf = tfidf.fit_transform(docs_list)
    query_tfidf = tfidf.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    docs['tfidf cosin'] = cosineSimilarities
    matched_doc = docs.loc[docs['tfidf cosin'].idxmax()][0]
    return matched_doc
