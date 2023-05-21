import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

#sample document
document = "I love playing cricket. Cricket is my favourite sport."

#Tokenization
sentences = sent_tokenize(document)
words = word_tokenize(document)

#pos tagging
pos_tags = pos_tag(words)

#stop words removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.casefold() not in stop_words]

#stemming 
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

#Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

#output
print('----------------------------------------------------------------------')
print('Original document : ', document)
print('----------------------------------------------------------------------')
print('Tokenization : ', words)
print('----------------------------------------------------------------------')
print('POS_tagging : ', pos_tags)
print('----------------------------------------------------------------------')
print('Stop words removal : ', filtered_words)
print('----------------------------------------------------------------------')
print('Stemming : ', stemmed_words)
print('----------------------------------------------------------------------')
print('Lemmatization : ', lemmatized_words)
print('----------------------------------------------------------------------')


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

d0 = 'jupiter is the largest planet'
d1 = 'Mars is the fourth planet from the sun'

string  = [d0, d1]
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(string)
print('----------------------------------------------------------------------')
print('Word indices : ', tfidf.vocabulary_)
print('TF_IDF values : ', result)
print('----------------------------------------------------------------------')

from sklearn.feature_extraction.text import CountVectorizer
tf = CountVectorizer()
result = tf.fit_transform(string)
print('Word indices : ', tf.vocabulary_)
print('TF values : ', result)
print('----------------------------------------------------------------------')
