import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string


doc1 = "The cat sat on my face"
doc2 = "The dog sat on my bed"

doc1_tokens= word_tokenize(doc1)
doc2_tokens= word_tokenize(doc2)

vocabulary = set(doc1_tokens).union(set(doc2_tokens))


wordDict1 = dict.fromkeys(vocabulary, 0) 
wordDict2 = dict.fromkeys(vocabulary, 0)

for word in doc1_tokens:
    wordDict1[word]+=1
    
for word in doc2_tokens:
    wordDict2[word]+=1
    
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict
    
tf1 = computeTF(wordDict1, doc1_tokens)
tf2 = computeTF(wordDict2, doc2_tokens)


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict
    
idfs = computeIDF([wordDict1, wordDict2])

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf
    
tfidfBow1 = computeTFIDF(tf1, idfs)
tfidfBow2 = computeTFIDF(tf2, idfs)

print(tfidfBow1)
print(tfidfBow2)

