# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def bow_extractor(corpus, ngram_range=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features  

def tfidf_extractor(corpus, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# 2016 300개의 영화에 대한 영화평 데이터를 사용해서 학습
with open('2016_filtered_review.txt', encoding='utf-8') as f:
    docs = [doc.strip().split('\t\t') for doc in f]
    docs = [(doc[1], int(doc[2])) for doc in docs if len(doc) == 3]
    # To read the second and third column info from each row
    texts, scores = zip(*docs)
    # 둘을 분리해서 별도의 list 변수로 저장

filtered_texts = []
filtered_labels = []

for text, score in zip(texts, scores):
    if 4 <= score <= 8:
        continue
        
    # 평점 기준으로 문서에 label을 부여
    # 1 ~ 3 -> 부정, -1
    # 9 ~ 10 -> 긍정, 1
    filtered_texts.append(text)
    filtered_labels.append(1 if score > 8 else -1)
    

num_reviews = len(filtered_texts) #788189

num_train = int(num_reviews*0.7) #551732
# 전체 리뷰 중에서 70%를 training data로 사용하고, 나머지 30%를 evaluation data로 사용
train_texts = filtered_texts[:num_train]
train_labels = filtered_labels[:num_train]
test_texts = filtered_texts[num_train+1:]
test_labels = filtered_labels[num_train+1:]

bow_vectorizer, train_bow_features = bow_extractor(train_texts)
test_bow_features = bow_vectorizer.transform(test_texts)
vocablist = [word for word, _ in sorted(bow_vectorizer.vocabulary_.items(), key=lambda x:x[1])]
# bow_vectorizer.vocabulary_.items() returns a list of (word, frequency)
# We sort words based on their frequencies and save the words

"""
tfidf_vectorizer, train_tfidf_features = tfidf_extractor(train_texts)
test_tfidf_features = tfidf_vectorizer.transform(test_texts)
"""

# Ridge regression
#lr = LogisticRegression(C=1000.0, random_state=0)
"""
Misclassified samples: 14350 out of 164522
Accuracy: 0.91
"""
lr = LogisticRegression(C=0.1, penalty='l1', random_state=0) # Lasso regression
# C = Inverse of regularization strength, 즉 C 값이 작을수록 penalty를 많이 준다는 것입니다.
# penalty를 많이 준다는 뜻은 L1 같은 경우는 feature의 수를 그만큼 많이 줄인다는 뜻이고
# L2인 경우는 weight 값을 더 0에 가깝게 한다는 뜻입니다.
"""
Misclassified samples: 11168 out of 164522
Accuracy: 0.93
"""
lr.fit(train_bow_features, train_labels)
pred_labels = lr.predict(test_bow_features)
print('Misclassified samples: {} out of {}'.format((pred_labels != test_labels).sum(),len(test_labels)))
print('Accuracy: %.2f' % accuracy_score(test_labels, pred_labels))

# Get coefficients of the model
coefficients = lr.coef_.tolist()

sorted_coefficients = sorted(enumerate(coefficients[0]), key=lambda x:x[1], reverse=True)
# 학습에 사용된 각 단어마다의 coefficient (즉 weight) 값이 존재
# coefficient값이 큰 순으로 정렬 'reverse=True'

print(sorted_coefficients[:5])
# print top 50 positive words
for word, coef in sorted_coefficients[:50]:
    print('{0:} ({1:.3f})'.format(vocablist[word], coef))
# print top 50 negative words
for word, coef in sorted_coefficients[-50:]:
    print('{0:} ({1:.3f})'.format(vocablist[word], coef))