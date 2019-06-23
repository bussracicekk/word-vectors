import sys,io
import numpy as np
from gensim.models import KeyedVectors
train_file = sys.argv[1]
infile = io.open(train_file,"r",encoding="utf-8-sig")

listword={}
capital = ""
for line in infile:
    if ':' in line:
        capital = line.replace(": ", "").replace("\n", "")
        listword[capital] = {}
    else:
        wordList = line.split(" ")
        listword[capital][line] = wordList

#b=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore')
#b.save('model')
model=KeyedVectors.load('model')


def find_cosine(vector1, vector2):
    product = vector1.dot(vector2)
    result = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2)))
    return product / result


def find_data(word1, word2, word3, model_word):
    max_num = -1000
    word1, word2, word3 = word1.lower(), word2.lower(), word3.lower()
    vector_result = model_word[word2] - model_word[word1] + model_word[word3]  # word3 - (word1 - word2)
    for word in model_word.vocab:
        #if word1 != diff_vec and word2 != diff_vec and word3 != diff_vec:
        vector = model_word[word]
        cos_sim = find_cosine(vector_result, vector)
        if cos_sim > max_num:
            max_num = cos_sim
            word4 = word

    return word4


def accuracy():
    accuracyDict={}
    for title, sentences in listword.items():
        count = 0
        for sentence in sentences.keys():
            if find_data(sentences[sentence][0], sentences[sentence][1], sentences[sentence][2], model.wv) == sentences[sentence][3]:
                count = count+1
        accuracyDict[title]=count/len(sentences.keys())
    return accuracyDict


print(accuracy())