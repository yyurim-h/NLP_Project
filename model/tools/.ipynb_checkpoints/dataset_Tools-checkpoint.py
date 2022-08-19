from kiwipiepy import Kiwi
import wikipedia
import re
import pandas as pd
from collections import defaultdict
import pickle
from konlpy.tag import Hannanum

stopwords = pd.read_table("C:/python/mrc_demomake/tools/korean100.txt", sep="\t")
stopwords = stopwords.iloc[:,0].tolist()
with open("C:/python/mrc_demomake/tools/inverted_index.pickle", "rb") as f:
    inverted_index = pickle.load(f)
#inverted_index = pd.read_pickle("inverted_index.pickle")

def __init__(self, string, steps, raw_data, asc_codes, docs_list, query, dataset, li):
    self.string = string
    self.steps = steps
    self.raw_data = raw_data
    self.asc_codes = asc_codes
    self.docs_list = docs_list
    self.query = query
    self.dataset = dataset
    self.li = li

def wiki_set(li):
    li2 = []
    wikipedia.set_lang("ko")
    text = str()
    for i in li:
        try:
            text = wikipedia.page(i).content
            text = re.sub(r'==.*?==+', '', text).replace('\n', '')
        except:
            pass
        li2.append(text)
    wiki_data = pd.Series(li2, name="document")
    wiki_data = wiki_data.drop_duplicates().reset_index().drop(columns="index")
    wiki_data = wiki_data.iloc[:,0].map(lambda x : re.sub(r'네이버캐스트', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r'<>', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r'<<>>', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r' <|> ', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r' 「」 ', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r' 「|」 ', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r' 『』 ', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r' 『|』 ', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r'\(([一-鿕]|[㐀-䶵]|[豈-龎])+\)', '', str(x)))
    wiki_data = wiki_data.map(lambda x : re.sub(r'[一-鿕]|[㐀-䶵]|[豈-龎]+', '', str(x)))
    return wiki_data
# 스트링 전처리(불용어)
def data_preprocessing(raw_data, asc_codes):
    raw_data = pd.Series(list(raw_data))
    raw_data= raw_data.map(lambda x : ord(x))
    for asc in asc_codes:
        raw_data.drop(raw_data[raw_data == asc].index, inplace = True)
    raw_data = raw_data.map(lambda x : chr(x))
    pre_data = "".join(raw_data.tolist())
    return pre_data
# 쿼리 인버트 인덱스로 만들기
def build_inverted_index(docs_list):
    invert_index = defaultdict(set)
    pos_tagger = Hannanum()
    for doc_id, docs in enumerate(docs_list):
        for word in pos_tagger.nouns(docs):
            if word not in stopwords:
                invert_index[word].add(doc_id)
    return invert_index
# 쿼리 인덱스 매기기
def process_and_search(query, dataset):
    matched_documents = set()
    pos_tagger = Hannanum()
    for word in pos_tagger.nouns(query):
        matches = inverted_index.get(word)
        if matches:
            matched_documents |= matches
            returned_documents = pd.DataFrame(dataset['documents'][list(matched_documents)])
    return returned_documents
# 3문장씩 나누기
def to_documents(string, steps):
    kiwi = Kiwi().split_into_sents(string)
    doc_li=[]
    for i in range(len(kiwi)-2):
        doc_li.append(str(kiwi[i][0])+str(kiwi[i+1][0])+str(kiwi[i+2][0]))
    return doc_li
