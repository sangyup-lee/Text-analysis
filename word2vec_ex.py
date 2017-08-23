# -*- coding: utf-8 -*-
"""
@author: Sang
"""

import sys
import os.path
import json
from gensim.models import Word2Vec


with open('2016_filtered_review.txt', encoding='utf-8') as f:
    docs = [doc.strip().split('\t\t') for doc in f]

docs1 = [doc[1].strip().split() for doc in docs]

print(docs1[:5])

model = Word2Vec(docs1, window=5, min_count=5, size=100)
# docs1 => list of list of words

for word, sim in model.similar_by_word(u"이정재", topn=20):
    print("{}\t{}".format(word, sim))
print('\n\n')

for word, sim in model.similar_by_word(u"예쁘", topn=20):
    print("{}\t{}".format(word, sim))
print('\n\n')

for word, sim in model.most_similar(positive=[u"이정재", u"이병헌"],negative=[u'황정민'],
                                    topn=20):
    print("{}\t{}".format(word, sim))
print('\n\n')

print(model.doesnt_match(u"이병헌 이정재 멋지".split()))
print('\n\n')

print(model.similarity(u'이병헌', u'이정재'))
print(model.similarity(u'이병헌', u'예쁘'))
print(model.wv.similarity(u'이병헌', u'예쁘'))
print(model.wv[u'이정재'])  # numpy vector of a word
